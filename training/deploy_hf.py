"""Deploy the Hugging Face side of SachAI: trained detector (private model repo) + ZeroGPU model Space.

    python training/deploy_hf.py            # uses HF_TOKEN (write) from env or backend/.env

Prints the value to put in HF_SPACE.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "backend" / ".env")
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
user = api.whoami()["name"]
model_repo, space_repo = f"{user}/sachai-detector", f"{user}/sachai-model-service"

api.create_repo(model_repo, private=True, exist_ok=True)
api.upload_folder(repo_id=model_repo, folder_path=ROOT / "training" / "model")
print(f"model  -> https://huggingface.co/{model_repo}")

# free accounts can only create Gradio Spaces directly on ZeroGPU hardware
if not api.repo_exists(space_repo, repo_type="space"):  # re-creating triggers the paid-hardware check
    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g")
api.add_space_secret(space_repo, "HF_TOKEN", token)  # lets the Space read the private model repo
api.add_space_variable(space_repo, "DETECTOR_REPO", model_repo)
api.upload_folder(repo_id=space_repo, repo_type="space", folder_path=ROOT / "space")
print(f"space  -> https://huggingface.co/spaces/{space_repo}")
print(f"\nHF_SPACE={space_repo}")
