import json
from functools import partial
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
import os
import numpy as np
from pipeline import add_context, concurrent_chunker
from utils import (split_pdf_by_title,
                   save_jsonl,
                   create_dense_index,
                   create_sparse_index,
                   init_pinecone,
                   dense_index_upsert,
                   sparse_index_upsert,
                   dense_index_query,
                   sparse_index_query)
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
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

    scores = [r["score"] for r in results]
    min_s, max_s = min(scores), max(scores)

    if max_s - min_s == 0:
        for r in results:
            r["normalized_score"] = 1.0
        return results

    for r in results:
        r["normalized_score"] = (r["score"] - min_s) / (max_s - min_s)
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
    print("eşleşmeler", final_ranked_list)
    return final_ranked_list


def create_rag_tool(
        pc,
        embed_model,
        vectorizer,
        dense_index_name = "rag-dense",
        sparse_index_name = "rag-sparse",
        top_k=5,
        alpha=0.5,
):
    def wrapped_hybrid_search(query):
        results = hybrid_search(
            query=query,
            pc=pc,
            embed_model=embed_model,
            vectorizer=vectorizer,
            dense_index_name=dense_index_name,
            sparse_index_name=sparse_index_name,
            top_k=top_k,
            alpha=alpha,
        )
        # agent'in işleyebilmesi için string olarak döndür
        return json.dumps(results, ensure_ascii=False)

    doc_search_tool = Tool(
        name="DocumentHybridSearch",
        func=wrapped_hybrid_search,
        description=(
            """
                    Kullanıcı şirket içi belgeler, teknik konular veya PDF'ler hakkında spesifik bir soru sorduğunda bu aracı kullan.
                    Genel bilgi, sohbet veya güncel hava durumu/haberler için kullanma.
                    Girdi olarak sadece kullanıcının sorgu metnini (string) alır.
                    """
        ),
    )
    return doc_search_tool

def rag_system_init(
        vectorizer_path="./sparse_vectorizer/",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        DENSE_INDEX_NAME="rag-dense",
        SPARSE_INDEX_NAME="rag-sparse"
):
    load_dotenv()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
    context_model_name = "meta-llama/Llama-3.2-1B-Instruct"
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
    with open(contexted_save_path, "r", encoding="utf-8") as f:
        for line in f:
            contexted_chunk = json.loads(line)
            contexted_text = contexted_chunk["chunk"]
            all_chunks_text.append(contexted_text)
            metadata.append(contexted_chunk)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        dtype=np.float64
    )
    vectorizer.fit(all_chunks_text)

    os.makedirs(vectorizer_path, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(vectorizer_path, "vectorizer.joblib"))

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
    return pc, embed_model, vectorizer


if __name__ == "__main__":
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pc, embed_model, vectorizer = rag_system_init()

    rag_tool = create_rag_tool(pc,
                               embed_model,
                               vectorizer)
    model_name = "gemini-2.0-flash"
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.5, google_api_key=google_api_key)

    print("Araç oluşturuluyor...")
    tools = [rag_tool]

    system_prompt = """
    Sen, belge tabanlı soruları yanıtlayan bir asistansın.
    Kullanıcı bir soru sorduğunda, görevin cevabı DocumentHybridSearch aracını 
    kullanarak bulmaktır. 
    Sana verilen JSON string'i (arama sonuçlarını) kullanarak kullanıcıya 
    doğal dilde bir cevap oluştur.
    Cevabı bilmediğini varsayma, HER ZAMAN önce arama aracını kullan.
    """

    # --- 3. AGENT'I OLUŞTURUN (PROMPT OLMADAN) ---
    print("Agent (çalıştırıcı) kuruluyor...")

    # create_agent fonksiyonu, 'prompt' olmadan çağrıldığında
    # ve model (Gemini) tool-calling'i desteklediğinde,
    # 'agent_runnable' zaten tam bir çalıştırıcı (executor) olur.
    agent_runnable = create_agent(llm, tools, system_prompt=system_prompt)

    # --- agent_executor = AgentExecutor(...) SATIRINI TAMAMEN SİLİN ---

    # --- 4. DOĞRUDAN 'agent_runnable'I ÇAĞIRIN ---
    print("Agent çalıştırılıyor...")
    query = "What is a theoretical framework for understanding the Scaled Dot-Product Attention (SDPA) mechanism?"

    # Çağırma formatı: "messages"
    result = agent_runnable.invoke({
        "messages": [
            ("user", query)
        ]
    })

    print("\n--- AGENT'IN NİHAİ CEVABI ---")
    # Çıktı formatı değişmiş olabilir, 'output' yerine 'messages' listesini kontrol edin
    if 'output' in result:
        print(result['output'])
    elif 'messages' in result and result['messages']:
        # Genellikle son mesaj (AIMessage) cevabı içerir
        print(result['messages'][-1].content)
    else:
        print(result)







