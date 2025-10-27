import json
import re
from typing import final
from functools import partial
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from pipeline import add_context, process_title, concurrent_chunker, hybrid_chunker
from utils import (split_pdf_by_title,
                   save_jsonl,
                   create_dense_index,
                   create_sparse_index,
                   init_pinecone,
                   dense_index_upsert,
                   sparse_index_upsert,
                   dense_index_query,
                   sparse_index_query)
from tqdm import tqdm
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download("punkt_tab")
from langchain_core.tools import Tool
import joblib

def normalize_scores(results):
    if not results:
        return []

    scores = [r["scores"] for r in results]
    min_s, max_s = min(scores), max(scores)

    if max_s - min_s == 0:
        for r in results:
            r["normalized_score"] = 1.0
        return results

    for r in results:
        r["normalized_score"] = (r["scores"] - min_s) / (max_s - min_s)
    return results

def hybrid_search(
        query,
        pc,
        embed_model,
        vectorizer,
        dense_index_name,
        sparse_index_name,
        top_k,
        alpha,
):
    dense_vector = embed_model.encode(query).tolist()
    sparse_matrix = vectorizer.transform([query])
    sparse_vector = {
        "indices":sparse_matrix.indices.tolist(),
        "values":sparse_matrix.data.tolist(),
    }

    dense_results = dense_index_query(pc, dense_index_name, dense_vector, top_k)
    sparse_results = sparse_index_query(pc, sparse_index_name, sparse_vector, top_k)

    dense_matches = normalize_scores(dense_results.get("matches", []))
    sparse_matches = normalize_scores(sparse_results.get("matches", []))

    all_results = {}
    for r in dense_matches:
        all_results[r["id"]] = {
            "dense_score":r["normalized_score"],
            "sparse_score":0.0,
            "metadata":r["metadata"]
        }

    for r in sparse_matches:
        if r["id"] in sparse_matches:
            all_results[r["id"]]["sparse_score"] = r["normalized_score"]
        else:
            all_results[r["id"]] = {
                "dense_score":0.0,
                "sparse_score":r["normalized_score"],
                "metadata":r["metadata"]
            }

    final_ranked_list = []
    for id, score in all_results.items():
        hybrid_score = alpha*score["dense_score"] + (1-alpha)*score["sparse_score"]
        final_ranked_list.append({
            "id":id,
            "hybrid_score":hybrid_score,
            "metadata":score["metadata"]
        })

    final_ranked_list.sort(key=lambda x:x["hybrid_score"], reverse=True)
    return final_ranked_list

def create_rag_tool(
        pc,
        embed_model,
        vectorizer,
        dense_index_name = "rag_dense",
        sparse_index_name = "rag_sparse",
        top_k=5,
        alpha=0.5,
):
    doc_search_func = partial(
        hybrid_search,
        pc = pc,
        embed_model = embed_model,
        vectorizer = vectorizer,
        dense_index_name = dense_index_name,
        sparse_index_name = sparse_index_name,
        top_k = top_k,
        alpha = alpha,
    )

    doc_search_tool = Tool(
        name="DocumentHybridSearch",
        func=doc_search_func,
        description="""
        Kullanıcı şirket içi belgeler, teknik konular veya PDF'ler hakkında spesifik bir soru sorduğunda bu aracı kullan. 
        Genel bilgi, sohbet veya güncel hava durumu/haberler için kullanma. 
        Girdi olarak sadece kullanıcının sorgu metnini (string) alır.
        """
    )

    return doc_search_tool

def rag_system_init(
        vectorizer_path="sparse_vectorizer",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        DENSE_INDEX_NAME="rag-dense",
        SPARSE_INDEX_NAME="rag-sparse"
):
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
    context_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    title_file_path = "saves/titles.jsonl"
    chunk_save_path = "saves/chunks.jsonl"
    contexted_save_path = "saves/contexts.jsonl"
    MAX_TOKENS = 200
    SIMILARITY_THRESHOLD = 0.5
    doc_dir_path = "docs"
    save_path = "saves/titles.jsonl"
    save_doc_path = "saves/docs.jsonl"
    processed_docs_dir = "processed_docs"
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    prompt_template = """
        Document
        <document>
        {doc}
        </document>

        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token)
    embed_model = SentenceTransformer(embed_model_name)
    dense_vector_dim = embed_model.get_sentence_embedding_dimension()
    generator = pipeline("text-generation",
                         model=context_model_name,
                         max_new_tokens=100,
                         temperature=0.7,
                         top_p=0.9,
                         truncation=True)


    pc = init_pinecone(PINECONE_API_KEY)
    create_dense_index(pc, DENSE_INDEX_NAME, dense_vector_dim)

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
            contexted_chunk = add_context(chunk, sections, generator, prompt_template)
            contexted_text = contexted_chunk['chunk']
            embedding = embed_model.encode(contexted_text, convert_to_numpy=True)
            dense_index_upsert(pc, DENSE_INDEX_NAME, [embedding], [contexted_chunk])
            save_jsonl([contexted_chunk], contexted_save_path)

    all_chunks_text = []
    metadata = []
    with open(contexted_save_path, "r") as f:
        for line in f:
            contexted_chunk = json.loads(line)
            contexted_text = contexted_chunk["chunk"]
            all_chunks_text.append(contexted_text)
            metadata.append(contexted_chunk)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        dtype=float
    )
    vectorizer.fit(all_chunks_text)

    if os.path.exists(vectorizer_path):
        shutil.rmtree(vectorizer_path)
    joblib.dump(vectorizer, vectorizer_path)

    create_sparse_index(pc, SPARSE_INDEX_NAME)
    sparse_vectors_batch = []
    batch_size = 100
    for i, chunk_text in enumerate(tqdm(all_chunks_text, desc="Sparse vektörler oluşturuluyor")):
        sparse_matrix = vectorizer.transform([chunk_text])
        indices = sparse_matrix.indices.tolist()
        values = sparse_matrix.data.tolist()

        sparse_vectors_batch.append({
            "id":f"chunk_{i}",
            "sparse_values":{
                "indices":indices,
                "values":values,
            },
            "metadata":metadata[i]
        })

        if len(sparse_vectors_batch) >= batch_size or i==len(all_chunks_text)-1:
            sparse_index_upsert(pc, SPARSE_INDEX_NAME, sparse_vectors_batch)
            sparse_vectors_batch = []

    print("Sparse vektörler başarıyla yüklendi")

