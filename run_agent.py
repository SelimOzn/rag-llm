import json
import sys

import joblib
from sentence_transformers import SentenceTransformer
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import (init_pinecone,
                   dense_index_query,
                   sparse_index_query,
                   config)


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
    return final_ranked_list


def create_rag_tool(
        pc,
        embed_model,
        vectorizer,
        dense_index_name = config.DENSE_INDEX_NAME,
        sparse_index_name = config.SPARSE_INDEX_NAME,
        top_k=5,
        alpha=0.5,
):
    def wrapped_hybrid_search(query: str):
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

def main():
    print("Agent başlatılıyor...")

    try:
        pc = init_pinecone(config.PINECONE_API_KEY)
        embed_model = SentenceTransformer(config.EMBED_MODEL_NAME)
        vectorizer = joblib.load(config.VECTORIZER_FILE_PATH)
    except FileNotFoundError:
        print(f"Hata: '{config.VECTORIZER_FILE_PATH}' bulunamadı.")
        print("Lütfen önce 'build_index.py' betiğini çalıştırarak indeksleri oluşturun.")
        return
    except Exception as e:
        print(f"Modeller yüklenirken bir hata oluştu: {e}")
        return

    llm = ChatGoogleGenerativeAI(model=config.AGENT_LLM_MODEL,
                                 temperature=0.5,
                                 google_api_key=config.GOOGLE_API_KEY)
    rag_tool = create_rag_tool(pc,embed_model,vectorizer)
    tools = [rag_tool]

    print("Agent kuruluyor...")
    agent_runnable = create_agent(llm, tools, system_prompt=config.AGENT_SYSTEM_PROMPT)

    print("Agent çalıştırılıyor... Çıkmak için 'exit' yazın.")
    while True:
        try:
            query = input("\nSorgunuz: ")
            if query.lower() == "exit":
                break

            result = agent_runnable.invoke({
                "messages":[
                    ("user",query)
                ]
            })

            if "messages" in result and result["messages"]:
                print(f"\nAgent: {result['messages'][-1].content}")
            else:
                print(f"\nAgent: {result.get('output', 'Bir cevap alınamadı.')}")

        except Exception as e:
            print(f"Bir hata oluştu: {e}")


if __name__ == "__main__":
    main()
