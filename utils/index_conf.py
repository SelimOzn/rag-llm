from pinecone import Pinecone, ServerlessSpec
import numpy as np


def init_pinecone(api_key: str):
    """
    Pinecone API key ile bağlantıyı başlatır.
    """
    pc = Pinecone(api_key=api_key)
    print("[✓] Pinecone initialized.")
    return pc


# -----------------------------
# 2️⃣ Indexleri Oluşturma
# -----------------------------
def create_dense_index(pc, dense_index_name, dense_vector_dim):

    if not pc.has_index(dense_index_name):
        print(f"[+] Creating dense index: {dense_index_name}")
        pc.create_index(
            name=dense_index_name,
            dimension=dense_vector_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="disabled",
        )
    else:
        print(f"[✓] Dense index already exists: {dense_index_name}")

def create_sparse_index(pc, sparse_index_name):
    if pc.has_index(sparse_index_name):
        print(f"Deleting current sparse index: {sparse_index_name}")
        pc.delete_index(sparse_index_name)
    print(f"[+] Creating sparse index: {sparse_index_name}")
    pc.create_index(
        name=sparse_index_name,
        vector_type="sparse",
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="disabled",
    )



# -----------------------------
# 3️⃣ Dense Index Upsert
# -----------------------------
def dense_index_upsert(pc, index_name, embeddings, metadatas):
    """
    Dense embeddingleri indexe upsert eder.
    embeddings: numpy array listesi
    metadatas: her embedding için dictionary
    """
    index = pc.Index(index_name)
    to_upsert = [
        (meta["chunk_id"], emb.tolist(), meta) for emb, meta in zip(embeddings, metadatas)
    ]
    index.upsert(vectors=to_upsert)
    print(f"[↑] Upserted {len(embeddings)} dense vectors.")


# -----------------------------
# 4️⃣ Sparse Index Upsert
# -----------------------------
def sparse_index_upsert(pc, index_name, sparse_vectors):
    """
    Sparse vectorleri indexe upsert eder.
    sparse_vectors: dict listesi {"indices": [...], "values": [...]}
    metadatas: her vector için dictionary
    """
    index = pc.Index(index_name)
    index.upsert(vectors=sparse_vectors)
    print(f"[↑] Upserted {len(sparse_vectors)} sparse vectors.")



def dense_index_query(pc, index_name, query_vector, top_k=5):
    index = pc.Index(index_name)
    result = index.query(vector=query_vector.tolist(), top_k=top_k)
    return result


def sparse_index_query(pc, index_name, sparse_query, top_k=5):
    index = pc.Index(index_name)
    result = index.query(sparse_query=sparse_query, top_k=top_k)
    return result