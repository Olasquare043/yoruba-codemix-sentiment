"""Upload the app/ folder to the Hugging Face Space (creates the Space if needed).

Needs HF_TOKEN (a write token) in a local .env file next to this repo's README, or in the environment.
Usage: python scripts/push_space.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

SPACE = "Olasquare/yoruba-tweet-sentiment"
ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not found in .env or the environment.")

api = HfApi(token=token)
api.create_repo(SPACE, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(folder_path=ROOT / "app", repo_id=SPACE, repo_type="space",
                  ignore_patterns=[".venv/*", "__pycache__/*", "*.pyc", ".env"],
                  commit_message="Update app")
print("Uploaded: https://huggingface.co/spaces/" + SPACE)
