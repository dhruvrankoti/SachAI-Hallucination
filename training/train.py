"""
Train the SachAI hallucination detector on Modal.

ModernBERT-base token classifier (LettuceDetect-style): given (context, answer) it
labels every answer token as supported (0) / hallucinated (1).

Data: RAGTruth (human span-level labels) + HaluEval QA/dialogue/summarization
(response-level labels -> whole hallucinated answer marked 1). Model selection and the
reported test score use RAGTruth only, so numbers stay comparable with the literature.

    modal run training/train.py                    # RAGTruth + HaluEval, H100, ~15 min
    modal run training/train.py --no-halueval      # RAGTruth only
    modal run training/train.py --smoke            # quick sanity run
"""
import modal

app = modal.App("sachai-train")
vol = modal.Volume.from_name("sachai-models", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "transformers==4.48.3", "accelerate==1.3.0", "requests", "numpy"
)

BASE = "answerdotai/ModernBERT-base"
MAX_LEN = 4096
RAW = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/"
HALUEVAL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/"


def load_ragtruth():
    import json, random, requests
    src = {}
    for line in requests.get(RAW + "source_info.jsonl").text.splitlines():
        s = json.loads(line)
        src[s["source_id"]] = s["prompt"]
    train, test = [], []
    for line in requests.get(RAW + "response.jsonl").text.splitlines():
        r = json.loads(line)
        ex = {"context": src[r["source_id"]], "answer": r["response"],
              "spans": [(l["start"], l["end"]) for l in r["labels"]]}
        (train if r["split"] == "train" else test).append(ex)
    random.Random(0).shuffle(train)
    n_dev = len(train) // 10
    return train[n_dev:], train[:n_dev], test


def load_halueval(n_summ: int = 3000):
    """Each item yields a faithful + a hallucinated answer for the same context."""
    import json, random, requests
    get = lambda f: [json.loads(l) for l in requests.get(HALUEVAL + f).text.splitlines() if l.strip()]
    items = []  # (context, faithful answer, hallucinated answer)
    for r in get("qa_data.json"):
        items.append((f"Question: {r['question']}\n\nContext:\n{r['knowledge']}", r["right_answer"], r["hallucinated_answer"]))
    for r in get("dialogue_data.json"):
        items.append((f"Question: {r['dialogue_history']}\n\nContext:\n{r['knowledge']}", r["right_response"], r["hallucinated_response"]))
    summ = get("summarization_data.json")
    random.Random(0).shuffle(summ)
    for r in summ[:n_summ]:
        items.append((f"Summarize the following news.\n\n{r['document']}", r["right_summary"], r["hallucinated_summary"]))
    data = []
    for ctx, good, bad in items:
        data += [{"context": ctx, "answer": good, "spans": []},
                 {"context": ctx, "answer": bad, "spans": [(0, len(bad))]}]
    return data


def encode(tok, ex):
    enc = tok(ex["context"], ex["answer"], truncation="only_first", max_length=MAX_LEN,
              return_offsets_mapping=True)
    labels = []
    for (s, e), seq in zip(enc["offset_mapping"], enc.sequence_ids()):
        if seq != 1 or s == e:
            labels.append(-100)
        else:
            labels.append(int(any(s < he and e > hs for hs, he in ex["spans"])))
    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc


@app.function(image=image, gpu="H100", volumes={"/models": vol}, timeout=2 * 3600)
def train(smoke: bool = False, epochs: int = 3, halueval: bool = True, out: str = "/models/detector-v2"):
    import json, numpy as np, torch
    from torch.utils.data import DataLoader
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              DataCollatorForTokenClassification, get_linear_schedule_with_warmup)

    tr, dev, test = load_ragtruth()
    if halueval:
        tr = tr + load_halueval()
    if smoke:
        tr, dev, test, epochs = tr[:200], dev[:50], test[:50], 1
    print(f"train={len(tr)} dev={len(dev)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(BASE, num_labels=2).cuda()
    coll = DataCollatorForTokenClassification(tok)
    enc = lambda xs: [encode(tok, x) for x in xs]
    tr_e, dev_e, test_e = enc(tr), enc(dev), enc(test)

    # length-bucketed batches -> far less padding -> faster + cheaper
    def batches(data, bs, shuffle):
        idx = np.argsort([len(d["input_ids"]) for d in data])
        chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        if shuffle:
            np.random.shuffle(chunks)
        for c in chunks:
            yield coll([data[i] for i in c])

    bs = 8
    steps = epochs * (len(tr_e) + bs - 1) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    @torch.no_grad()
    def predict(data):
        model.eval()
        preds = []  # per example: (probs, labels) over answer tokens
        for b in batches(data, 16, False):
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float().softmax(-1)[..., 1]
            for pi, li in zip(p.cpu().numpy(), b["labels"].cpu().numpy()):
                m = li != -100
                preds.append((pi[m], li[m]))
        model.train()
        return preds

    def metrics(preds, th):
        tp = fp = fn = etp = efp = efn = 0
        for p, l in preds:
            h = p > th
            tp += int((h & (l == 1)).sum()); fp += int((h & (l == 0)).sum()); fn += int((~h & (l == 1)).sum())
            ph, lh = bool(h.any()), bool((l == 1).any())
            etp += ph and lh; efp += ph and not lh; efn += lh and not ph
        f1 = lambda a, b, c: 2 * a / max(2 * a + b + c, 1)
        return {"token_f1": round(f1(tp, fp, fn), 4), "example_f1": round(f1(etp, efp, efn), 4),
                "example_precision": round(etp / max(etp + efp, 1), 4), "example_recall": round(etp / max(etp + efn, 1), 4)}

    best, best_th = -1, 0.5
    model.train()
    for ep in range(epochs):
        for i, b in enumerate(batches(tr_e, bs, True)):
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if i % 200 == 0:
                print(f"ep{ep} step{i} loss={loss.item():.4f}", flush=True)
        dp = predict(dev_e)
        th, m = max(((t, metrics(dp, t)) for t in np.arange(0.05, 0.9, 0.05)), key=lambda x: x[1]["example_f1"])
        print(f"epoch {ep} dev @th={th:.2f}: {m}", flush=True)
        if m["example_f1"] > best:
            best, best_th = m["example_f1"], float(th)
            model.save_pretrained(out); tok.save_pretrained(out)

    model = AutoModelForTokenClassification.from_pretrained(out).cuda()
    res = {"threshold": round(best_th, 2), "data": "ragtruth+halueval" if halueval else "ragtruth", "dev_example_f1": best, "test": metrics(predict(test_e), best_th)}
    print("FINAL", res)
    json.dump(res, open(f"{out}/sachai.json", "w"))
    vol.commit()
    return res


@app.local_entrypoint()
def main(smoke: bool = False, epochs: int = 3, halueval: bool = True, out: str = "/models/detector-v2"):
    print(train.remote(smoke=smoke, epochs=epochs, halueval=halueval, out=out))
