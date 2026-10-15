import os
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
manifest_dir = os.path.join(project_root, "data", "manifests")
lm_dir = os.path.join(project_root, "lm")

os.makedirs(lm_dir, exist_ok=True)

corpus_path = os.path.join(lm_dir, "text_corpus.txt")

with open(corpus_path, "w", encoding="utf-8") as fout:
    # Thêm cả train_360.csv
    for split in ["train.csv", "train_360.csv", "dev.csv", "test.csv"]:
        path = os.path.join(manifest_dir, split)
        if os.path.exists(path):
            df = pd.read_csv(path)
            for text in df["text"]:
                fout.write(text.lower().strip() + "\n")

print(f"✅ Saved corpus: {corpus_path}")

# Tạo corpus dạng char-level
with open(corpus_path, "r", encoding="utf-8") as fin, \
     open(os.path.join(lm_dir, "text_corpus_char.txt"), "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip().lower()
        fout.write(" ".join(list(line)) + "\n")