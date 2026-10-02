"""
SachAI backend — RAG answer generation with hallucination detection and self-correcting retries.

Flow for /ask:
  retrieve context (uploaded docs via Postgres full-text search, else Wikipedia + BM25)
  -> cross-encoder rerank (HF ZeroGPU Space) -> LLM answer (Groq / OpenRouter)
  -> detector (HF ZeroGPU Space, ModernBERT trained on RAGTruth) [+ numeric guard, + Jev]
  -> if hallucinated: feed flagged spans back and regenerate (max MAX_RETRIES)
  -> log the run to Postgres (real usage stats at /stats)
"""
import asyncio
import io
import json
import math
import os
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

# Answer models the user can pick from. Only models whose provider key is set are offered.
PROVIDERS = {  # name -> (OpenAI-compatible endpoint, API key)
    "groq": ("https://api.groq.com/openai/v1/chat/completions", os.environ.get("GROQ_API_KEY")),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", os.environ.get("OPEN_ROUTER_API_KEY")),
}
MODELS = {  # id -> (label, provider, provider's model id)
    "gpt-oss-120b": ("GPT-OSS 120B", "groq", "openai/gpt-oss-120b"),
    "gpt-oss-20b": ("GPT-OSS 20B", "groq", "openai/gpt-oss-20b"),
    "qwen3.8-27b": ("Qwen3.8 27B", "groq", "qwen/qwen3.8-27b"),
    "llama-3.3-70b": ("Llama 3.3 70B", "openrouter", "meta-llama/llama-3.3-70b-instruct"),
    "gemini-2.5-flash": ("Gemini 2.5 Flash", "openrouter", "google/gemini-2.5-flash"),
    "claude-haiku-4.5": ("Claude Haiku 4.5", "openrouter", "anthropic/claude-haiku-4.5"),
    "deepseek-v3.1": ("DeepSeek V3.1", "openrouter", "deepseek/deepseek-chat-v3.1"),
    "gpt-4o-mini": ("GPT-4o mini", "openrouter", "openai/gpt-4o-mini"),
}
AVAILABLE = [m for m, (_, p, _) in MODELS.items() if PROVIDERS[p][1]]
DEFAULT_MODEL = AVAILABLE[0] if AVAILABLE else None
HELPER_MODEL = "gpt-oss-20b" if "gpt-oss-20b" in AVAILABLE else DEFAULT_MODEL  # fast model for internal steps
HF_SPACE = os.environ.get("HF_SPACE")  # e.g. "user/sachai-model-service"
HF_TOKEN = os.environ.get("HF_TOKEN")  # ZeroGPU quota is charged to this account
DATABASE_URL = os.environ.get("DATABASE_URL")
TYPESAFE_KEY = os.environ.get("TYPESAFE_API_KEY")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 3))
TOP_K = 5

SCHEMA = """
create table if not exists documents (
  id bigserial primary key, session_id text not null, name text not null, created_at timestamptz default now());
create table if not exists chunks (
  id bigserial primary key, doc_id bigint not null references documents(id) on delete cascade,
  session_id text not null, title text not null, text text not null,
  tsv tsvector generated always as (to_tsvector('english', text)) stored);
create index if not exists chunks_tsv on chunks using gin (tsv);
create index if not exists chunks_session on chunks (session_id);
create table if not exists queries (
  id bigserial primary key, session_id text, question text not null, model text, mode text, answer text,
  verified boolean, refused boolean, attempts int, verdicts jsonb, latency_ms int, created_at timestamptz default now());
alter table queries add column if not exists model text;
"""
db: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(_):
    global db
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    # Neon URLs carry channel_binding, which asyncpg would forward as an (invalid) server setting;
    # statement cache off so Neon's pooled (PgBouncer) endpoint works too
    url = urlsplit(DATABASE_URL)
    query = urlencode([(k, v) for k, v in parse_qsl(url.query) if k != "channel_binding"])
    db = await asyncpg.create_pool(urlunsplit(url._replace(query=query)), min_size=1, max_size=5, statement_cache_size=0)
    await db.execute(SCHEMA)
    yield
    await db.close()


app = FastAPI(title="SachAI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(filter(None, [os.environ.get("FRONTEND_URL"), "http://localhost:5173", "http://127.0.0.1:5173"])),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
http = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10), follow_redirects=True, headers={"User-Agent": "SachAI/1.0 (https://github.com/sachai-hallucination) httpx"})

# ---------------------------------------------------------------- retrieval

STOP = set("a an the of to in on for and or is are was were be by with as at from that this it its what which who how why when do does did".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def chunk_text(text: str, title: str, size: int = 160, overlap: int = 40) -> list[dict]:
    """~size-word chunks that keep line breaks (tables and lists stay readable), with ~overlap words carried over."""
    lines = []
    for line in filter(str.strip, text.splitlines()):
        words = line.split()
        lines += [" ".join(words[i:i + size]) for i in range(0, len(words), size)]  # split over-long lines
    chunks, cur = [], []
    for line in lines:
        if cur and sum(len(l.split()) for l in cur) + len(line.split()) > size:
            chunks.append(cur)
            keep = []
            for l in reversed(cur):
                if sum(len(k.split()) for k in keep) + len(l.split()) > overlap:
                    break
                keep.insert(0, l)
            cur = keep
        cur.append(line)
    if cur:
        chunks.append(cur)
    return [{"title": title, "text": "\n".join(c)} for c in chunks]


def bm25(query: str, chunks: list[dict], k: int = 30) -> list[dict]:
    docs = [tokenize(c["text"]) for c in chunks]
    q = tokenize(query)
    if not docs or not q:
        return chunks[:k]
    avg = sum(map(len, docs)) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)

    def score(d):
        tf = Counter(d)
        return sum(math.log(1 + (n - df[t] + .5) / (df[t] + .5)) * tf[t] * 2.2 / (tf[t] + 1.2 * (.25 + .75 * len(d) / avg))
                   for t in q if t in tf)

    ranked = sorted(zip(map(score, docs), range(n)), reverse=True)
    return [chunks[i] for _, i in ranked[:k]]


_space = None


def _space_predict(*args, api_name: str):
    global _space
    if not HF_SPACE:
        raise HTTPException(500, "HF_SPACE not configured")
    if _space is None:
        from gradio_client import Client
        _space = Client(HF_SPACE, hf_token=HF_TOKEN, verbose=False)
    return _space.predict(*args, api_name=api_name)


async def space(*args, api_name: str):
    """Call the model service (HF ZeroGPU Space) without blocking the event loop."""
    return await asyncio.to_thread(_space_predict, *args, api_name=api_name)


def jev_nouls(state: str, instructions: list[str]) -> list[float]:
    """One Jev call answering several yes/no questions about `state` (~0.3 s)."""
    from typesafe_sdk import Noul, TypeSafeClient
    r = TypeSafeClient().system_one(state=state, questions={f"q{i}": Noul(instructions=t) for i, t in enumerate(instructions)})
    return [float(r.answers[f"q{i}"].noul) for i in range(len(instructions))]


async def rerank(query: str, chunks: list[dict], k: int = TOP_K) -> list[dict]:
    """Jev scores every candidate passage in one call; without Jev, keep the lexical (BM25 / FTS) order."""
    if len(chunks) <= k or not TYPESAFE_KEY:
        return chunks[:k]
    state = f"QUESTION: {query}\n\n" + "\n\n".join(f"PASSAGE {i}: {c['text']}" for i, c in enumerate(chunks))
    try:
        scores = await asyncio.to_thread(jev_nouls, state, [f"PASSAGE {i} contains facts needed to answer the QUESTION or to check whether it is true"
                                                            for i in range(len(chunks))])
    except Exception:
        return chunks[:k]
    return [c for _, c in sorted(zip(scores, chunks), key=lambda x: -x[0])][:k]


WIKI = "https://en.wikipedia.org/w/api.php"
PAGE_CACHE: dict[str, list[dict]] = {}  # title -> chunks


async def wiki_page(title: str) -> list[dict]:
    # full-text extracts come back one page per request, so pages are fetched in parallel
    if title not in PAGE_CACHE:
        p = (await http.get(WIKI, params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                          "titles": title, "format": "json"})).json()
        page = next(iter(p.get("query", {}).get("pages", {}).values()), {})
        if "missing" in page or not page.get("extract"):
            return []
        if len(PAGE_CACHE) > 500:
            PAGE_CACHE.clear()
        PAGE_CACHE[title] = [{**c, "url": f"https://en.wikipedia.org/?curid={page['pageid']}"}
                             for c in chunk_text(page["extract"], page["title"])]
    return PAGE_CACHE[title]


async def wiki_search(query: str, n: int = 2) -> list[str]:
    s = (await http.get(WIKI, params={"action": "query", "list": "search", "srsearch": query, "srlimit": n, "format": "json"})).json()
    return [p["title"] for p in s.get("query", {}).get("search", [])]


async def suggest_titles(question: str) -> list[str]:
    """Wikipedia search isn't semantic, so ask the LLM which articles likely hold the answer."""
    try:
        out = "".join([t async for t in llm_stream([{"role": "user", "content":
            f"Question or claim: {question}\nList up to 3 English Wikipedia article titles most likely to contain the facts "
            "needed to answer it or check whether it is true. "
            "One title per line, nothing else."}], 0.0, max_tokens=200, model=HELPER_MODEL)])
        return [t.strip("-*• \"'") for t in out.splitlines() if t.strip()][:3]
    except Exception:
        return []


async def wikipedia(question: str) -> list[dict]:
    """Two parallel tracks: LLM-suggested titles -> fetch, and plain search -> fetch. ~1-1.5 s total."""
    async def track(titles_coro):
        return await asyncio.gather(*map(wiki_page, await titles_coro))

    pages = [p for t in await asyncio.gather(track(suggest_titles(question)), track(wiki_search(question))) for p in t]
    seen, chunks = set(), []
    for p in pages:
        if p and p[0]["title"] not in seen:
            seen.add(p[0]["title"])
            chunks += p
    return chunks


async def search_docs(question: str, session_id: str, k: int = 30) -> list[dict]:
    """Postgres full-text search over the session's uploaded chunks (any query term may match)."""
    terms = " | ".join(tokenize(question))
    if not terms:
        return []
    rows = await db.fetch(
        """select title, text from chunks where session_id = $1 and tsv @@ to_tsquery('english', $2)
           order by ts_rank_cd(tsv, to_tsquery('english', $2)) desc limit $3""", session_id, terms, k)
    return [dict(r) for r in rows]


async def retrieve(question: str, session_id: str) -> tuple[str, list[dict]]:
    web = asyncio.create_task(wikipedia(question))  # start speculatively: hides the DB round-trip
    if await db.fetchval("select exists(select 1 from chunks where session_id = $1)", session_id):
        web.cancel()
        return "docs", await rerank(question, await search_docs(question, session_id, k=20))
    return "web", await rerank(question, bm25(question, await web, k=20))


def build_context(sources: list[dict]) -> str:
    # "passage N:" mirrors RAGTruth's format, which the detector was trained on
    return "\n\n".join(f"passage {i + 1}:{s['title']}: {s['text']}" for i, s in enumerate(sources))

# ---------------------------------------------------------------- generation

REFUSAL = "I couldn't find this in the provided sources."
DIRECT = "You are SachAI, a precise assistant. Answer concisely and correctly."


async def needs_sources(question: str) -> bool:
    """Jev (~0.3 s): does answering require outside facts? Math, logic and small talk don't. Defaults to True."""
    if not TYPESAFE_KEY:
        return True
    try:
        [no_facts] = await asyncio.to_thread(jev_nouls, f"QUESTION: {question}", [
            "The QUESTION can be answered completely by calculation, logic, or casual conversation, "
            "without needing any factual knowledge about the world, people, places, events, or documents"])
        return no_facts < 0.5
    except Exception:
        return True
SYSTEM = ("You are SachAI, a precise assistant. Use ONLY facts explicitly stated in the context. "
          "Do not add outside knowledge, guesses, or details not in the context. Be concise and direct. "
          "If the user asks a question, answer it. If the user makes a statement, say whether the context shows it is "
          "true or false, and state the correct facts from the context. "
          f"If the context does not contain the needed facts, reply exactly: \"{REFUSAL}\"")


async def llm_stream(messages: list[dict], temperature: float, max_tokens: int = 700, model: str | None = None):
    model = model or DEFAULT_MODEL
    if model not in AVAILABLE:
        raise HTTPException(500, "No LLM configured: set GROQ_API_KEY or OPEN_ROUTER_API_KEY")
    _, provider, model_id = MODELS[model]
    url, key = PROVIDERS[provider]
    body = {"model": model_id, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True}
    if provider == "openrouter":
        body["provider"] = {"sort": "throughput"}
    if "gpt-oss" in model_id:
        body["reasoning_effort"] = "low"  # reasoning tokens count toward max_tokens and add latency
    async with http.stream("POST", url, json=body, headers={"Authorization": f"Bearer {key}"}) as r:
        if r.status_code != 200:
            raise HTTPException(502, f"LLM error {r.status_code}: {(await r.aread())[:200]!r}")
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                choices = json.loads(line[6:]).get("choices") or [{}]
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta

# ---------------------------------------------------------------- verification


def numeric_guard(answer: str, haystack: str) -> list[dict]:
    """Flag multi-digit numbers in the answer that never appear in context/question."""
    hay = haystack.replace(",", "")
    return [{"start": m.start(), "end": m.end(), "text": m.group(), "prob": 1.0, "by": "numeric"}
            for m in re.finditer(r"\d[\d,.]*\d%?", answer) if m.group().replace(",", "").rstrip("%") not in hay]


JEV_THRESHOLD = 0.2  # a sentence below this Jev support score is flagged; tuned on RAGTruth dev (training/results/jev.json)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    bounds, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+", text):
        bounds.append((start, m.start()))
        start = m.end()
    bounds.append((start, len(text)))
    return [(a, b) for a, b in bounds if len(text[a:b].strip()) > 2] or [(0, len(text))]


def jev_sentences(context: str, answer: str) -> tuple[list[dict], float]:
    """Jev scores every answer sentence for support in one call. Returns (flagged spans, lowest support)."""
    sents = sentence_spans(answer)
    state = f"CONTEXT:\n{context}\n\nCLAIMS:\n" + "\n".join(f"{i}. {answer[a:b]}" for i, (a, b) in enumerate(sents))
    nouls = jev_nouls(state, [f"Claim {i} is fully and explicitly supported by the CONTEXT (every detail, no additions)"
                              for i in range(len(sents))])
    spans = [{"start": a, "end": b, "text": answer[a:b], "prob": round(1 - n, 3), "by": "jev"}
             for (a, b), n in zip(sents, nouls) if n < JEV_THRESHOLD]
    return spans, min(nouls)


def with_numeric(spans: list[dict], answer: str, haystack: str) -> list[dict]:
    extra = [n for n in numeric_guard(answer, haystack) if not any(s["start"] <= n["start"] < s["end"] for s in spans)]
    return sorted(spans + extra, key=lambda s: s["start"])


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().replace("\u2019", "'") == REFUSAL.lower()


REFUSED = {"hallucinated": False, "score": 0.0, "spans": [], "jev": None, "refused": True}


async def detector_check(context: str, question: str, answer: str) -> dict:
    """Trained ModernBERT detector on the HF Space (most accurate, ~2-4 s)."""
    det = await space(context, question, answer, api_name="/detect")
    spans = with_numeric(det["spans"], answer, context + question)
    return {"hallucinated": bool(spans), "score": det["score"], "spans": spans, "by": "detector"}


async def verify_fast(context: str, question: str, answer: str) -> dict:
    """Used inside the /ask retry loop: Jev (~0.3 s). Falls back to the detector without a Jev key."""
    if is_refusal(answer):  # an honest "not found" is never a hallucination
        return REFUSED
    if not TYPESAFE_KEY:
        return {**await detector_check(context, question, answer), "jev": None}
    spans, support = await asyncio.to_thread(jev_sentences, context, answer)
    spans = with_numeric(spans, answer, context + question)
    return {"hallucinated": bool(spans), "score": round(1 - support, 3), "spans": spans, "jev": support, "by": "jev"}


async def verify_deep(context: str, question: str, answer: str) -> dict:
    """Used by /verify: detector verdict, with Jev's support score alongside when configured."""
    if is_refusal(answer):
        return REFUSED
    det, jev = await asyncio.gather(
        detector_check(context, question, answer),
        asyncio.to_thread(jev_sentences, context, answer) if TYPESAFE_KEY else asyncio.sleep(0),
        return_exceptions=True)
    if isinstance(det, Exception):
        raise HTTPException(502, f"Detector unavailable: {det}")
    return {**det, "jev": jev[1] if isinstance(jev, tuple) else None}

# ---------------------------------------------------------------- API


class AskIn(BaseModel):
    question: str
    session_id: str = ""
    model: str = ""


class VerifyIn(BaseModel):
    context: str
    answer: str
    question: str = ""


def ev(**kw) -> str:
    return json.dumps(kw) + "\n"


@app.post("/ask")
async def ask(body: AskIn):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Question is required")
    model = body.model or DEFAULT_MODEL
    if model not in AVAILABLE:
        raise HTTPException(400, f"Model not available: {model}")

    async def run():
        t0, mode, attempts, final = time.perf_counter(), None, [], None
        try:
            retrieval = asyncio.create_task(retrieve(q, body.session_id))  # starts now; routing takes ~0.3 s
            if not await needs_sources(q):  # e.g. "what is 2+2?": nothing to look up or ground
                retrieval.cancel()
                mode = "direct"
                yield ev(type="sources", mode=mode, sources=[])
                yield ev(type="attempt", n=0)
                answer = ""
                async for tok in llm_stream([{"role": "system", "content": DIRECT}, {"role": "user", "content": q}], 0.0, model=model):
                    answer += tok
                    yield ev(type="token", n=0, text=tok)
                attempts.append((answer, {"hallucinated": False, "spans": [], "direct": True}))
                final = {"answer": answer, "verified": False, "refused": False}
                yield ev(type="final", best=0, direct=True, **final)
                return
            mode, sources = await retrieval
            yield ev(type="sources", mode=mode, sources=sources)
            if not sources:
                yield ev(type="final", answer="I couldn't find any relevant sources for this question.", verified=False, best=0)
                return
            context = build_context(sources)
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Context:\n{context}\n\nUser: {q}"}]
            for n in range(MAX_RETRIES + 1):
                yield ev(type="attempt", n=n)
                answer = ""
                async for tok in llm_stream(messages, 0.2 if n == 0 else 0.0, model=model):
                    answer += tok
                    yield ev(type="token", n=n, text=tok)
                v = await verify_fast(context, q, answer)
                attempts.append((answer, v))
                yield ev(type="verdict", n=n, **v)
                if not v["hallucinated"]:
                    break
                flagged = "\n".join(f'- "{s["text"]}"' for s in v["spans"]) or "- (overall answer judged unsupported)"
                messages += [{"role": "assistant", "content": answer},
                             {"role": "user", "content": "A fact-checker found these parts of your answer are NOT supported by the context:\n"
                              f"{flagged}\n\nRewrite the full answer using only facts explicitly stated in the context. "
                              "Remove or correct the unsupported parts. Output only the new answer."}]
            best = min(range(len(attempts)), key=lambda i: sum(s["end"] - s["start"] for s in attempts[i][1]["spans"]) + attempts[i][1]["hallucinated"])
            final = {"answer": attempts[best][0], "verified": not attempts[best][1]["hallucinated"],
                     "refused": attempts[best][1].get("refused", False)}
            yield ev(type="final", best=best, **final)
            # second opinion from the trained detector; streamed after the answer so it never delays it
            if HF_SPACE and TYPESAFE_KEY and not final["refused"]:
                try:
                    audit = await asyncio.wait_for(detector_check(context, q, final["answer"]), timeout=20)
                    yield ev(type="audit", **audit)
                except Exception:
                    yield ev(type="audit", unavailable=True)
        except HTTPException as e:
            yield ev(type="error", message=e.detail)
        except Exception as e:  # surface to UI instead of a broken stream
            yield ev(type="error", message=str(e))
        if final:
            try:
                await db.execute(
                    """insert into queries (session_id, question, model, mode, answer, verified, refused, attempts, verdicts, latency_ms)
                       values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                    body.session_id, q, model, mode, final["answer"], final["verified"], final["refused"], len(attempts),
                    json.dumps([v for _, v in attempts]), int((time.perf_counter() - t0) * 1000))
            except Exception as e:  # logging must never break an answer
                print(f"[db] failed to log query: {e}")

    return StreamingResponse(run(), media_type="application/x-ndjson")


@app.post("/verify")
async def verify_route(body: VerifyIn):
    if not body.context.strip() or not body.answer.strip():
        raise HTTPException(400, "Both context and answer are required")
    return await verify_deep(body.context, body.question, body.answer)


async def doc_names(session_id: str) -> list[str]:
    return [r["name"] for r in await db.fetch("select name from documents where session_id = $1 order by id", session_id)]


@app.post("/upload")
async def upload(session_id: str = Form(...), files: list[UploadFile] = File(...)):
    for f in files:
        data = await f.read()
        if f.filename.lower().endswith(".pdf"):
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
        else:
            text = data.decode("utf-8", errors="ignore")
        if not text.strip():
            raise HTTPException(422, f"No readable text in {f.filename}")
        text = text.replace("\x00", "")  # Postgres text can't hold NUL bytes
        async with db.acquire() as con, con.transaction():
            doc_id = await con.fetchval("insert into documents (session_id, name) values ($1, $2) returning id", session_id, f.filename)
            await con.executemany("insert into chunks (doc_id, session_id, title, text) values ($1, $2, $3, $4)",
                                  [(doc_id, session_id, c["title"], c["text"]) for c in chunk_text(text, f.filename)])
    return {"docs": await doc_names(session_id)}


@app.get("/files/{session_id}")
async def list_docs(session_id: str):
    return {"docs": await doc_names(session_id)}


@app.delete("/files/{session_id}")
async def clear_docs(session_id: str):
    await db.execute("delete from documents where session_id = $1", session_id)
    return {"docs": []}


@app.get("/models")
def models():
    return {"default": DEFAULT_MODEL, "models": [{"id": m, "label": MODELS[m][0], "provider": MODELS[m][1]} for m in AVAILABLE]}


@app.get("/stats")
async def stats():
    """Live usage numbers, computed from every logged /ask run."""
    r = await db.fetchrow("""
        select count(*) as questions,
               count(*) filter (where verified and not refused) as verified,
               count(*) filter (where refused) as refused,
               count(*) filter (where attempts > 1) as needed_retry,
               count(*) filter (where attempts > 1 and verified) as fixed_by_retry,
               round(avg(attempts), 2)::float as avg_attempts,
               round(avg(latency_ms))::int as avg_latency_ms
        from queries""")
    return dict(r)


@app.get("/health")
async def health():
    """Also wakes the model Space so the first question doesn't pay its cold start."""
    if HF_SPACE:
        try:
            await http.get(f"https://{HF_SPACE.replace('/', '-').replace('_', '-').lower()}.hf.space", timeout=5)
        except Exception:
            pass
    return {"ok": True, "jev": bool(TYPESAFE_KEY), "max_retries": MAX_RETRIES}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
