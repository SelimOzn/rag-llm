import json

def add_context(chunk, doc, model, prompt_template):
    doc_id = chunk['doc_id']
    prompt = prompt_template.format(doc=doc, chunk=chunk['chunk'])
    contexted_text = model(prompt, max_length=200)[0]["generated_text"]
    contexted_chunk = {"doc_id":doc_id,
                       "title":chunk['title'],
                       "chunk_id":chunk['chunk_id'],
                       "chunk":contexted_text}

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