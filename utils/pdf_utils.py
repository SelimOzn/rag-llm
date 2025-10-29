import fitz
import json
import os
from .io_utils import save_jsonl

def split_pdf_by_title(pdf_path, save_path, save_doc_path, doc_id):
    with fitz.open(pdf_path) as doc:
        sections = []
        doc_text = []
        current_section = {"title":"First", "doc_id":f"doc_{doc_id}", "content":[]}
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

                        doc_text.append(text)

                        font_flags = span["flags"]
                        is_bold = font_flags & 16 != 0
                        has_gaps = has_surrounding_whitespace(lines, i)
                        if has_gaps and is_bold:
                            if current_section["content"]:
                                current_section["content"] = "".join(current_section["content"])
                                sections.append(current_section)
                            else:
                                text = current_section["title"] + " " + text
                            current_section = {"title":text, "doc_id":f"doc_{doc_id}", "content":[]}
                        else:
                            current_section["content"].append(text)

        if current_section["content"]:
            current_section["content"] = " ".join(current_section["content"])
            sections.append(current_section)

        save_jsonl(sections, save_path)

        full_doc = {
            "doc_id":f"doc_{doc_id}",
            "doc_text":" ".join(doc_text)
        }

        save_jsonl([full_doc], save_doc_path)

    return sections, full_doc

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



if __name__ == "__main__":
    doc_dir_path = "../docs"
    save_path = "../saves/titles.jsonl"
    save_doc_path = "../saves/docs.jsonl"
    for i, file in enumerate(os.listdir(doc_dir_path)):
        doc_path = os.path.join(doc_dir_path, file)
        print(doc_path)
        sections = split_pdf_by_title(doc_path, save_path, save_doc_path, i)
        for section in sections:
            print("Başlık: ", section["title"])
            print("Metin: ", "".join(section["content"][:5]), "...")
            print("-"*40)

