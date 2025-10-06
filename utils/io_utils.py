from pathlib import Path
import json


def save_jsonl(data, path):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
