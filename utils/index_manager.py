from utils import config
import os
import shutil
import json
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from pipeline import add_context, concurrent_chunker, add_context_in_batch
from utils import (split_pdf_by_title,
                   save_jsonl,
                   create_dense_index,
                   create_sparse_index,
                   dense_index_upsert,
                   sparse_index_upsert)


def run_rebuild(pc, tokenizer, embed_model, generator):

    print("Veri işleme ve indeksleme (tam yeniden oluşturma) başlıyor...")
    os.makedirs(config.DOC_DIR_PATH, exist_ok=True)
    os.makedirs(config.PROCESSED_DOCS_DIR, exist_ok=True)
    os.makedirs(config.SAVES_DIR, exist_ok=True)
    os.makedirs(config.VECTORIZER_DIR_PATH, exist_ok=True)
    os.makedirs(config.PROCESSED_DOCS_DIR, exist_ok=True)
    os.makedirs(config.SAVES_DIR, exist_ok=True)

    print("PDF'ler işleniyor ve dense index oluşturuluyor...")
    dense_vector_dim = embed_model.get_sentence_embedding_dimension()
    create_dense_index(pc, config.DENSE_INDEX_NAME, dense_vector_dim)
    doc_files = [f for f in os.listdir(config.DOC_DIR_PATH) if f.endswith('.pdf')]
    doc_id_counter = -1
    for doc_file in tqdm(doc_files, desc="PDF İşleme"):
        doc_id_counter+=1
        doc_path = os.path.join(config.DOC_DIR_PATH, doc_file)
        sections, doc = split_pdf_by_title(doc_path,
                                           config.TITLE_SAVE_PATH,
                                           config.DOC_SAVE_PATH,
                                           doc_id_counter)
        chunks = concurrent_chunker(sections,
                                    config.CHUNK_SAVE_PATH,
                                    emb_model=embed_model,
                                    tokenizer=tokenizer,
                                    max_tokens=config.MAX_TOKENS,
                                    similarity_threshold=config.SIMILARITY_THRESHOLD)

        metadatas, contexted_texts = add_context_in_batch(chunks,
                                                          sections,
                                                          generator,
                                                          config.CONTEXT_PROMPT_TEMPLATE,
                                                          config.CONTEXT_BATCH_SIZE)
        if not contexted_texts:
            print(f"Hiçbir chunk işlenemedi (muhtemelen doküman eşleşme hatası).")
            continue

        print(f"TOPLU embedding hesaplanıyor ({len(contexted_texts)} adet)...")

        embeddings = embed_model.encode(contexted_texts, convert_to_numpy=True)
        print("TOPLU dense index yüklemesi yapılıyor...")
        dense_index_upsert(pc, config.DENSE_INDEX_NAME, embeddings, metadatas)
        print("TOPLU .jsonl kaydı yapılıyor...")
        save_jsonl(metadatas, config.CONTEXTED_SAVE_PATH)
        shutil.move(doc_path, os.path.join(config.PROCESSED_DOCS_DIR, doc_file))

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
        curr_metadata = metadata[i]
        sparse_vectors_batch.append({
            "id": curr_metadata["chunk_id"],
            "sparse_values": {
                "indices": indices,
                "values": values,
            },
            "metadata": curr_metadata
        })

        if len(sparse_vectors_batch) >= config.SPARSE_BATCH_SIZE or i == len(all_chunks_text) - 1:
            sparse_index_upsert(pc, config.SPARSE_INDEX_NAME, sparse_vectors_batch)
            sparse_vectors_batch = []

    print("İndeksleme tamamlandı. Sparse vektör başarıyla yüklendi.")
