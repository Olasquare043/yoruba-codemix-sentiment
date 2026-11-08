"""Download the AfriSenti-SemEval 2023 Yoruba files (CC BY 4.0) into data/raw/."""
import os, urllib.request
BASE = "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data/yor"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(out, exist_ok=True)
for split in ["train", "dev", "test"]:
    dest = os.path.join(out, f"{split}.tsv")
    urllib.request.urlretrieve(f"{BASE}/{split}.tsv", dest)
    print("saved", dest)
