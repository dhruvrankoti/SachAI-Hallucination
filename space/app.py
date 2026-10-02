"""
SachAI model service — Hugging Face ZeroGPU Space (Gradio).

API (called by the backend through gradio_client):
    /detect  (context, question, answer) -> {hallucinated, score, threshold, spans}
"""
import json
import os

import gradio as gr
import spaces
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForTokenClassification, AutoTokenizer

DETECTOR = os.environ.get("DETECTOR_REPO", "sachai/sachai-detector")
TOKEN = os.environ.get("HF_TOKEN")  # needed if the detector repo is private
# ZeroGPU: models go on cuda at import time; a real GPU is attached only inside @spaces.GPU calls.
# Off Spaces without a GPU (local testing) everything runs on CPU.
DEVICE = "cuda" if os.environ.get("SPACE_ID") or torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(DETECTOR, token=TOKEN)
model = AutoModelForTokenClassification.from_pretrained(
    DETECTOR, token=TOKEN, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32).to(DEVICE).eval()
cfg = os.path.join(DETECTOR, "sachai.json") if os.path.isdir(DETECTOR) else hf_hub_download(DETECTOR, "sachai.json", token=TOKEN)
THRESHOLD = json.load(open(cfg))["threshold"]


def ragtruth_prompt(context: str, question: str) -> str:
    # the detector was trained on RAGTruth's prompt templates and is far more precise with them
    if question:
        return (f"Briefly answer the following question:\n{question}\nBear in mind that your response should be "
                f"strictly based on the following passages:\n{context}\n\nIn case the passages do not contain the "
                "necessary information to answer the question, please reply with: \"Unable to answer based on given passages.\"\noutput:")
    return f"Summarize the following news:\n{context}\n\noutput:"


@spaces.GPU(duration=20)
def detect(context: str, question: str, answer: str) -> dict:
    enc = tok(ragtruth_prompt(context, question), answer, truncation="only_first", max_length=4096,
              return_offsets_mapping=True, return_tensors="pt")
    offsets, seq = enc.pop("offset_mapping")[0].tolist(), enc.sequence_ids()
    with torch.no_grad():
        probs = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits.float().softmax(-1)[0, :, 1].tolist()
    spans, max_p = [], 0.0
    for (s, e), sq, p in zip(offsets, seq, probs):
        if sq != 1 or s == e:
            continue
        max_p = max(max_p, p)
        if p <= THRESHOLD:
            continue
        if spans and s - spans[-1]["end"] <= 1:  # merge adjacent tokens
            spans[-1]["end"], spans[-1]["prob"] = e, max(spans[-1]["prob"], p)
        else:
            spans.append({"start": s, "end": e, "prob": p})
    spans = [s for s in spans if s["end"] - s["start"] >= 3]  # drop single-char noise
    for s in spans:
        s["text"], s["prob"] = answer[s["start"]:s["end"]], round(s["prob"], 3)
    return {"hallucinated": bool(spans), "score": round(max_p, 3), "threshold": THRESHOLD, "spans": spans}


with gr.Blocks(title="SachAI model service") as demo:
    gr.Markdown("# SachAI model service\nHallucination detector (ModernBERT fine-tuned on RAGTruth). API only.")
    c, q, a = gr.Textbox(label="Context"), gr.Textbox(label="Question"), gr.Textbox(label="Answer")
    gr.Button("Detect").click(detect, [c, q, a], gr.JSON(), api_name="detect")

demo.queue(default_concurrency_limit=4).launch()
