import csv
import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"\b[a-z]+\b")
SPACE_RE = re.compile(r"\s+")
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    text = TAG_RE.sub(" ", text.lower())
    tokens = TOKEN_RE.findall(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [STEMMER.stem(t) for t in tokens]
    return SPACE_RE.sub(" ", " ".join(tokens)).strip()


def main():
    dataset_dir = Path("aclImdb")
    output_dir = Path("processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        rows = []
        for label_name, label in [("pos", 1), ("neg", 0)]:
            folder = dataset_dir / split / label_name
            for file_path in sorted(folder.glob("*.txt")):
                raw = file_path.read_text(encoding="utf-8", errors="ignore")
                rows.append((preprocess(raw), label))

        out_file = output_dir / f"{split}.csv"
        with out_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerows(rows)



if __name__ == "__main__":
    main()
