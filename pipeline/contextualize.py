import json

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





"""
import json

def add_context(chunk_file_path, doc, model, prompt_template):
    contexted_chunks = []
    with open(chunk_file_path, 'r') as file:
        chunks = [json.loads(line) for line in file]
        for chunk in chunks:
            doc_id = chunk['doc_id']
            prompt = prompt_template.format(doc=doc, chunk=chunk['chunk'])
            contexted_text = model.generate(prompt=prompt)
            contexted_chunk = {"doc_id":doc_id,
                               "title":chunk['title'],
                               "chunk_id":chunk['chunk_id'],
                               "chunk":contexted_text}
            contexted_chunks.append(contexted_chunk)

    return contexted_chunks
"""