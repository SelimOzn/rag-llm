import fitz
from sympy.physics.units import current
from torch.ao.nn.quantized.functional import threshold
import json
from pathlib import Path

def split_pdf_by_title(pdf_path, save_path):
    with fitz.open(pdf_path) as doc:
        sections = []
        current_section = {"title":"First", "content":[]}
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue
                lines = block["lines"]
                for i, line in enumerate(lines):
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        font_flags = span["flags"]
                        is_bold = font_flags & 16 != 0
                        has_gaps = has_surrounding_whitespace(lines, i)
                        if has_gaps and is_bold:
                            if current_section["content"]:
                                current_section["content"] = "".join(current_section["content"])
                                sections.append(current_section)
                            else:
                                text = current_section["title"] + " " + text
                            current_section = {"title":text, "content":[]}
                        else:
                            current_section["content"].append(text)

        if current_section["content"]:
            current_section["content"] = " ".join(current_section["content"])
            sections.append(current_section)

        save_jsonl(sections, save_path)

    return sections

def has_surrounding_whitespace(lines, i, threshold=5):
    current_y0, current_y1 = lines[i]["bbox"][1], lines[i]["bbox"][3]
    if i > 0:
        prev_y0, prev_y1 = lines[i-1]["bbox"][1], lines[i-1]["bbox"][3]
        top_gap = current_y0 - prev_y1
    else:
        top_gap=999

    if i < len(lines)-1:
        next_y0, next_y1 = lines[i+1]["bbox"][1], lines[i+1]["bbox"][3]
        bottom_gap = next_y0 - current_y1
    else:
        bottom_gap=999

    return top_gap > threshold or bottom_gap > threshold

def save_jsonl(data, path):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    doc_path = "docs/1706.03762v7.pdf"
    save_path = "saves/titles.jsonl"
    sections = split_pdf_by_title(doc_path, save_path)

    for section in sections:
        print("Başlık: ", section["title"])
        print("Metin: ", "".join(section["content"][:5]), "...")
        print("-"*40)


