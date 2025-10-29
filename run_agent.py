import json
import joblib
from sentence_transformers import SentenceTransformer
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import (init_pinecone,
                   dense_index_query,
                   sparse_index_query,
                   config,
                   normalize_scores,
                   hybrid_search)
# Terminalde çalışan uygulama. Buradan fonksiyon kullanma
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
