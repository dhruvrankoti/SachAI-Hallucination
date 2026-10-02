"""
Benchmark TypeSafe Jev vs the SachAI detector on RAGTruth (response-level hallucination detection).

- Jev: one call per response, one Noul ("is this sentence supported?") per answer sentence;
  response is flagged if any sentence scores below a threshold tuned on a dev sample.
- Detector: training/model with its own dev-tuned threshold (sachai.json), run on CPU.
- Dev = 200 random responses from RAGTruth's train split; test = 400 random from the official test split.

    TYPESAFE_API_KEY=... python training/bench_jev.py      # writes training/results/jev.json
"""
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typesafe_sdk import Noul, TypeSafeClient

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / "backend" / ".env")
RAW = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/"


def load(n_dev=200, n_test=400):
    src = {json.loads(l)["source_id"]: json.loads(l)["prompt"] for l in requests.get(RAW + "source_info.jsonl").text.splitlines()}
    rows = [json.loads(l) for l in requests.get(RAW + "response.jsonl").text.splitlines()]
    ex = lambda r: {"context": src[r["source_id"]], "answer": r["response"], "label": bool(r["labels"])}
    rng = random.Random(0)
    train = [ex(r) for r in rows if r["split"] == "train"]
    test = [ex(r) for r in rows if r["split"] == "test"]
    return rng.sample(train, n_dev), rng.sample(test, n_test)


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s) > 2] or [text]


jev = TypeSafeClient()


def jev_score(e):
    sents = sentences(e["answer"])
    t = time.perf_counter()
    r = jev.system_one(
        state=f"CONTEXT:\n{e['context']}\n\nCLAIMS:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(sents)),
        questions={f"s{i}": Noul(instructions=f"Claim {i} is fully and explicitly supported by the CONTEXT (every detail, no additions)")
                   for i in range(len(sents))})
    return min(a.noul for a in r.answers.values()), time.perf_counter() - t


tok = AutoTokenizer.from_pretrained(ROOT / "model")
model = AutoModelForTokenClassification.from_pretrained(ROOT / "model").eval()
TH = json.load(open(ROOT / "model" / "sachai.json"))["threshold"]


@torch.no_grad()
def detector_pred(e):
    enc = tok(e["context"], e["answer"], truncation="only_first", max_length=4096, return_tensors="pt")
    p = model(**enc).logits.softmax(-1)[0, :, 1]
    mask = torch.tensor([s == 1 for s in enc.sequence_ids()])
    return bool((p[mask] > TH).any())


def prf(pred, gold):
    tp = sum(p and g for p, g in zip(pred, gold)); fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"f1": round(2 * pr * rc / max(pr + rc, 1e-9), 4), "precision": round(pr, 4), "recall": round(rc, 4)}


if __name__ == "__main__":
    dev, test = load()
    with ThreadPoolExecutor(8) as pool:
        dev_j = list(pool.map(jev_score, dev))
        test_j = list(pool.map(jev_score, test))
    gold_dev, gold_test = [e["label"] for e in dev], [e["label"] for e in test]
    th = max(np.arange(0.05, 0.96, 0.05), key=lambda t: prf([s < t for s, _ in dev_j], gold_dev)["f1"])
    lat = [l for _, l in test_j]
    det = [detector_pred(e) for e in test]
    res = {
        "test_samples": len(test), "dev_samples": len(dev), "positives_in_test": sum(gold_test),
        "jev": {"threshold": round(float(th), 2), **prf([s < th for s, _ in test_j], gold_test),
                "latency_p50_s": round(float(np.median(lat)), 3), "latency_p90_s": round(float(np.percentile(lat, 90)), 3)},
        "detector": prf(det, gold_test),
        "jev_or_detector": prf([(s < th) or d for (s, _), d in zip(test_j, det)], gold_test),
        "jev_and_detector": prf([(s < th) and d for (s, _), d in zip(test_j, det)], gold_test),
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open(ROOT / "results" / "jev.json", "w"), indent=2)
