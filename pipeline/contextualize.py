import json
from tqdm import tqdm
import gc
import torch

def add_context(chunk, docs, generator, prompt_template):
    # doc_id ve title ile eşleşen belgeyi bul
    matched_doc = next(
        (doc for doc in docs if doc["doc_id"] == chunk["doc_id"] and doc["title"] == chunk["title"]),
        None
    )

    if matched_doc is None:
        raise ValueError(f"No matching document found for {chunk['doc_id']} - {chunk['title']}")

    # sadece o belgenin content'ini al
    doc_content = matched_doc["content"].strip()
    chunk_text = chunk["chunk"].strip()

    # promptu oluştur
    prompt = [
        {"role": "user", "content": prompt_template.format(doc=doc_content, chunk=chunk_text)}
    ]

    output = generator(prompt)
    full_text = output[0]["generated_text"]
    # prompt kısmını atarak sadece cevabı al
    answer = full_text[-1]["content"].strip()

    # yeni context'li chunk'ı döndür
    contexted_chunk = {
        "doc_id": chunk["doc_id"],
        "title": chunk["title"],
        "chunk_id": chunk["chunk_id"],
        "chunk": answer + chunk_text
    }

    return contexted_chunk


def add_context_in_batch(chunks, docs, generator, prompt_template, batch_size=4):

    doc_lookup = {}
    for doc in docs:
        if doc["doc_id"] not in doc_lookup:
            doc_lookup[doc["doc_id"]] = {}
        doc_lookup[doc["doc_id"]][doc["title"]] = doc["content"].strip()

    prompts_to_process = []
    valid_chunks = []

    print(f"Toplu işlem için {len(chunks)} adet prompt hazırlanıyor...")
    for chunk in chunks:
        try:
            doc_content = doc_lookup[chunk["doc_id"]][chunk["title"]]
            chunk_text = chunk["chunk"].strip()

            prompt_text = prompt_template.format(doc=doc_content, chunk=chunk_text)
            prompts_to_process.append([{
                "role":"user",
                "content": prompt_text
            }])
            valid_chunks.append(chunk)
        except KeyError:
            print(f"Uyarı: Eşleşen doküman bulunamadı {chunk['doc_id']} - {chunk['title']}. Bu chunk atlanıyor.")
            continue

    print(f"Generator'a {len(prompts_to_process)} prompt gönderiliyor (batch_size={batch_size})...")
    outputs = []

    for i in tqdm(range(0,len(prompts_to_process), batch_size), desc="Context Batch İşleme"):
        batch_outputs = generator(prompts_to_process[i:i+batch_size])
        outputs.extend(batch_outputs)
        del batch_outputs
        gc.collect()
        torch.cuda.empty_cache()

    contexted_chunks = []
    contexted_chunk_texts = []

    for chunk, output in zip(valid_chunks, outputs):
        #answer = output["generated_text"][-1]["content"].strip()
        answer = output[0]["generated_text"][-1]["content"].strip()
        contexted_text = answer + " " + chunk["chunk"].strip()
        contexted_chunks.append({
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "chunk_id": chunk["chunk_id"],
            "chunk": contexted_text
        })
        contexted_chunk_texts.append(contexted_text)

    print(f"{len(contexted_chunks)} adet chunk başarıyla işlendi.")
    return contexted_chunks, contexted_chunk_texts

