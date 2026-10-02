# SachAI: Grounded Answers That Correct Their Own Hallucinations

LLMs state made-up facts with full confidence. SachAI answers questions from real sources, checks every sentence of its own answer against those sources, and rewrites the answer when something isn't supported. In our measured runs, a verified answer arrives in 3–4 seconds.

## How it works

```
question
  │
  ├─ 0. Route ────── Jev (~0.3 s): does this need outside facts? "what is 2+2?" / "hi" → answered directly (~1–2 s),
  │                  labeled "No sources needed". Everything else continues below.
  │
  ├─ 1. Retrieve ─── your uploaded docs (Postgres full-text search)
  │                  or Wikipedia (LLM-suggested titles ∥ keyword search, pages fetched in parallel)
  │                  → Jev scores 20 candidate passages in one call → top 5
  │
  ├─ 2. Generate ─── the LLM you pick (Groq / OpenRouter) answers using only those passages, streamed live
  │
  ├─ 3. Verify ───── Jev checks every sentence against the passages (one call, ~0.3 s)
  │                  + a numeric guard (numbers that appear nowhere in the sources are flagged)
  │                  │
  │                  ├─ unsupported? → the flagged sentences go back to the LLM: "rewrite without these" → retry (max 3)
  │                  └─ supported?   → verified answer shown
  │
  └─ 4. Deep check ─ our trained hallucination detector audits the final answer in the background
                     and highlights any phrase it flags (never delays the answer)
```

- **Why two checkers?** Jev is fast enough to run inside the retry loop. Our trained detector is slightly more accurate but needs a GPU round-trip, so it audits the final answer. Both were benchmarked on the same data (below).
- **Questions or claims:** ask a question, or state a claim ("Mark Zuckerberg is the CEO of Amazon") and SachAI says whether the sources show it's true or false, with the correct facts.
- **Routing check:** on 12 test questions it routed 11 correctly. The miss was a deliberately mixed question ("What is 2+2 in Roman numerals history?"), which was still answered correctly.
- **Honest refusals:** if the sources don't contain the answer, the model is told to say so. A refusal is never counted as a hallucination.
- **Fallback:** if all retries still contain unsupported claims, the most grounded attempt is shown with the unsupported parts highlighted.
- **Logging:** every run is stored in Postgres. `/stats` and the "How it works" page show live numbers from those logs.

## The trained detector

The detector is a ModernBERT-base token classifier fine-tuned on **RAGTruth**, a human-annotated benchmark of real LLM answers with span-level hallucination labels. Given (sources, answer), it marks each answer token as supported or hallucinated.

- **Training:** [`training/train.py`](training/train.py), run on Modal (A100).
- **Splits:** we used RAGTruth's official train split, holding out 10% (1,509) to choose the threshold and epoch, which leaves 13,581 for training.
- **Raw results:** [`training/results/`](training/results).

**Results on the RAGTruth official test split (2,700 responses):**

| Model | Train data | Threshold | Dev example F1 | Test example F1 | Precision | Recall | Token F1 |
|---|---|---|---|---|---|---|---|
| **v1 (deployed)** | RAGTruth | 0.30 | 0.801 | **0.787** | 0.783 | 0.791 | 0.579 |
| v2 | RAGTruth + HaluEval | 0.15 | 0.795 | 0.743 | 0.679 | 0.821 | 0.548 |

- **Example-level:** does the answer contain any hallucinated token (predicted vs. gold)? **Token-level:** per-token F1 on answer tokens.
- These metrics come from our own evaluation code, not RAGTruth's official script, so token F1 isn't directly comparable to span-level numbers in papers. Each is a single run with no fixed seed, and contexts are truncated to 4,096 tokens.
- Adding HaluEval hurt. Its hallucinated answers are ChatGPT-generated and labeled per response, not per span, so v2 learned to over-flag.

## Jev vs. the trained detector

Both were run on the same 400 RAGTruth test responses, 132 of which contain hallucinations. The script is [`training/bench_jev.py`](training/bench_jev.py) and raw output is in [`training/results/jev.json`](training/results/jev.json).

Jev scores each answer sentence (one call per response), and a response is flagged if any sentence falls below 0.2. That threshold was tuned on 200 responses from RAGTruth's train split.

| Verifier | F1 | Precision | Recall | Latency |
|---|---|---|---|---|
| Trained detector (v1) | **0.754** | 0.722 | 0.788 | 2–4 s via ZeroGPU Space |
| Jev | 0.734 | 0.647 | **0.849** | **0.32 s p50 / 0.36 s p90** |
| Detector AND Jev | 0.750 | **0.802** | 0.705 | |
| Detector OR Jev | 0.739 | 0.612 | 0.932 | |

On this subset the detector scores 0.754, versus 0.787 on the full test set. Jev is 0.02 F1 behind and about 10× faster.

## Latency

These are measured end-to-end times for 5 questions. The backend ran on a laptop in India and called the live services (Neon, Wikipedia, Groq gpt-oss-120b, Jev, HF Space). Latency from Render has not been measured.

| Step | Time |
|---|---|
| Retrieval (Wikipedia + Jev rerank) | 2.2–2.5 s |
| First answer token | 2.7–3.1 s |
| **Verified final answer** | **3.1–3.8 s** |
| Deep check from the trained detector | +1.8–2.2 s after the answer (about 9 s on a cold Space) |

## Repo

```
backend/     FastAPI app (app.py): retrieval, generation, verify-and-retry loop, Postgres
frontend/    React + Vite + Tailwind UI: Ask, Verify, How it works
space/       Hugging Face ZeroGPU Space that serves the trained detector
training/    train.py (detector training), bench_jev.py (benchmark), deploy_hf.py, results/
render.yaml  Render blueprint for the backend
```

## Answer models

Users pick the model in the UI. A model only appears when its provider key is set:

- **Groq** (`GROQ_API_KEY`): GPT-OSS 120B (default), GPT-OSS 20B, Qwen3.8 27B
- **OpenRouter** (`OPEN_ROUTER_API_KEY`): Llama 3.3 70B, Gemini 2.5 Flash, Claude Haiku 4.5, DeepSeek V3.1, GPT-4o mini

## Deploy (free tiers)

| Part | Host | Notes |
|---|---|---|
| Frontend | Vercel | Static Vite build |
| Backend | Render (free web service) | Sleeps after 15 min idle, about 1 min to wake |
| Detector | Hugging Face ZeroGPU Space | 5 GPU-min/day on a free account. Needs an account 30+ days old with a verified email |
| Database | Neon Postgres (free) | Uploaded docs and chunks (full-text indexed), plus a log of every `/ask` run |

1. **Detector (Hugging Face):**
   ```bash
   python training/deploy_hf.py   # needs HF_TOKEN (write) in backend/.env and the trained model in training/model
   ```
   This pushes the model to a private repo, creates the ZeroGPU Space, and prints the `HF_SPACE` value.
2. **Database:** create a Neon project and copy its connection string. Tables are created on first start.
3. **Backend (Render):** choose New → Blueprint → this repo (`render.yaml`). Set `DATABASE_URL`, `HF_SPACE`, `HF_TOKEN`, `TYPESAFE_API_KEY`, `GROQ_API_KEY` and/or `OPEN_ROUTER_API_KEY`, and `FRONTEND_URL` (your Vercel URL).
4. **Frontend (Vercel):** import the repo with root directory `frontend`, and set `VITE_API_URL` to your Render URL.

Without `TYPESAFE_API_KEY` the app still works: retrieval keeps the lexical ranking, and the slower trained detector drives the retry loop.

## Run locally

```bash
cd backend && pip install -r requirements.txt && cp .env.example .env   # fill in keys
python app.py                                                             # http://localhost:8000
cd frontend && npm install && npm run dev                                 # http://localhost:5173
```

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/ask` | `{question, session_id, model?}` | NDJSON stream: `sources`, `attempt`, `token`, `verdict`, `final`, `audit` |
| POST | `/verify` | `{context, answer, question?}` | `{hallucinated, score, spans[], jev}` (trained detector) |
| POST | `/upload` | multipart `session_id`, `files[]` (pdf/txt/md) | `{docs}` |
| GET / DELETE | `/files/{session_id}` | none | `{docs}` |
| GET | `/models` | none | Answer models available to pick |
| GET | `/stats` | none | Live counts from the `queries` table |
| GET | `/health` | none | Health check; also wakes the detector Space |
