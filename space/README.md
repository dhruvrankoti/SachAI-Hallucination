---
title: SachAI Model Service
emoji: ✅
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
---

Hallucination detector (ModernBERT-base fine-tuned on RAGTruth), served on ZeroGPU for the SachAI backend.
Set the Space hardware to **ZeroGPU** and add secrets `HF_TOKEN` (read access to the model repo) and `DETECTOR_REPO`.
