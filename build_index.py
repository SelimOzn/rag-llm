from utils import config
import os
import shutil
import json
import numpy as np
import joblib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from pipeline import add_context, concurrent_chunker
from utils import (split_pdf_by_title,
                   save_jsonl,
                   create_dense_index,
                   create_sparse_index,
                   init_pinecone,
                   dense_index_upsert,
                   sparse_index_upsert)

def build_vector_stores():
    print("Veri işleme ve indeksleme başlıyor...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    embed_model = SentenceTransformer(config.EMBED_MODEL_NAME)
    dense_vector_dim = embed_model.get_sentence_embedding_dimension()
    generator = pipeline("text-generation",
                             model=config.CONTEXT_MODEL_NAME,
                             max_new_tokens=100,
                             temperature=0.7,
                             top_p=0.9,
                             truncation=True)

    pc = init_pinecone(config.PINECONE_API_KEY)
    create_dense_index(pc, config.DENSE_INDEX_NAME, dense_vector_dim)

    os.makedirs(config.PROCESSED_DOCS_DIR, exist_ok=True)
    os.makedirs(config.SAVES_DIR, exist_ok=True)

    print("PDF'ler işleniyor ve dense index oluşturuluyor...")
    doc_files = [f for f in os.listdir(config.PROCESSED_DOCS_DIR) if f.endswith('.pdf')]
    for i,file in enumerate(tqdm(doc_files, desc="PDF İşleme")):
        doc_path = os.path.join(config.PROCESSED_DOCS_DIR, file)
        sections, doc = split_pdf_by_title(doc_path, config.TITLE_SAVE_PATH, config.DOC_SAVE_PATH, i)
        chunks = concurrent_chunker(sections,
                                    config.CHUNK_SAVE_PATH,
                                    emb_model=embed_model,
                                    tokenizer=tokenizer,
                                    max_tokens=config.MAX_TOKENS,
                                    similarity_threshold=config.SIMILARITY_THRESHOLD)

        for chunk in chunks:
            contexted_chunk = add_context(chunk, sections, generator, config.CONTEXT_MODEL_NAME)
            contexted_text = contexted_chunk['chunk']
            embedding = embed_model.encode(contexted_text, convert_to_numpy=True)
            dense_index_upsert(pc, config.DENSE_INDEX_NAME, [embedding], [contexted_chunk])
            save_jsonl([contexted_chunk], config.CONTEXTED_SAVE_PATH)

        shutil.move(doc_path, os.path.join(config.PROCESSED_DOCS_DIR, file))
    print("TF-IDF Vectorizer eğitiliyor...")
    all_chunks_text = []
    metadata = []
    with open(config.CONTEXTED_SAVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            contexted_chunk = json.loads(line)
            all_chunks_text.append(contexted_chunk["chunk"])
            metadata.append(contexted_chunk)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        dtype=np.float64
    )
    vectorizer.fit(all_chunks_text)
    os.makedirs(config.VECTORIZER_DIR_PATH, exist_ok=True)
    joblib.dump(vectorizer, config.VECTORIZER_FILE_PATH)
    print(f"Vectorizer '{config.VECTORIZER_FILE_PATH}' olarak kaydedildi.")

    print("Sparse index oluşturuluyor ve vektörler yükleniyor...")
    create_sparse_index(pc, config.SPARSE_INDEX_NAME)
    sparse_vectors_batch = []

    for i, chunk_text in enumerate(tqdm(all_chunks_text, desc="Sparse Vektör Yükleme")):
        sparse_matrix = vectorizer.transform([chunk_text])
        indices = sparse_matrix.indices.tolist()
        values = sparse_matrix.data.tolist()

        sparse_vectors_batch.append({
            "id":metadata[i]["chunk_id"],
            "sparse_values":{
                "indices":indices,
                "values":values,
            },
            "metadata":metadata[i]
        })

        if len(sparse_vectors_batch) >= config.SPARSE_BATCH_SIZE or i==len(all_chunks_text)-1:
            sparse_index_upsert(pc, config.SPARSE_INDEX_NAME, sparse_vectors_batch)
            sparse_vectors_batch = []

    print("İndeksleme tamamlandı. Sparse vektör başarıyla yüklendi.")

if __name__ == "__main__":
    import nltk
    nltk.download("punkt_tab")
    build_vector_stores()




