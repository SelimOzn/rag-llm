import json

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from pipeline import add_context, process_title, concurrent_chunker, hybrid_chunker
from utils import split_pdf_by_title, has_surrounding_whitespace, save_jsonl
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rank_bm25 import BM25Okapi


if __name__ == '__main__':
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
    context_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    title_file_path = "saves/titles.jsonl"
    chunk_save_path = "saves/chunks.jsonl"
    contexted_save_path = "saves/contexts.jsonl"
    MAX_TOKENS = 200
    SIMILARITY_THRESHOLD = 0.5
    doc_dir_path = "docs"
    save_path = "saves/titles.jsonl"
    save_doc_path = "saves/docs.jsonl"
    processed_docs_dir = "processed_docs"

    prompt_template = """
    Here is the chunk we want to situate within the document:

    Chunk:
    {chunk}
    
    Document context (optional surrounding text to help situate chunk):
    {doc}
    
    Write 1-2 sentences explaining the role or position of this chunk within the document for better search retrieval. 
    Answer concisely and only provide the context.
"""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token)
    embed_model = SentenceTransformer(embed_model_name)
    context_tokenizer = AutoTokenizer.from_pretrained(context_model_name, token=token)
    context_model = AutoModelForCausalLM.from_pretrained(context_model_name, token=token)
    generator = pipeline("text-generation", model=context_model, tokenizer=context_tokenizer)
    bm25_vectorizer = TfidfVectorizer()

    for i, file in enumerate(os.listdir(doc_dir_path)):
        doc_path = os.path.join(doc_dir_path, file)
        sections, doc = split_pdf_by_title(doc_path, save_path, save_doc_path, i)
        shutil.move(doc_path, os.path.join(processed_docs_dir, file))
        chunks = concurrent_chunker(sections,
                                    chunk_save_path,
                                    emb_model=embed_model,
                                    tokenizer=tokenizer,
                                    max_tokens=MAX_TOKENS,
                                    similarity_threshold=SIMILARITY_THRESHOLD)

        for chunk in chunks:
            contexted_chunk = add_context(chunk["chunk"], doc, generator, prompt_template)
            save_jsonl(contexted_chunk, contexted_save_path)

    all_chunks = []
    with open(contexted_save_path, "r") as f:
        for line in f:
            contexted_chunk = json.loads(line)
            all_chunks.extend(contexted_chunk["chunk"].lower().split())
    bm25 = BM25Okapi(all_chunks)








