import json
import joblib
import sys
import shutil
from sentence_transformers import SentenceTransformer
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, pipeline
import os
import gradio as gr
from typing import Dict, Any, Optional
from utils import (init_pinecone,
                   config,
                   hybrid_search,
                   run_rebuild)

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

print("Global state (modeller, agent) yükleniyor... Lütfen bekleyin.")
GLOBAL_STATE: Dict[str, Any] = {}

try:
    GLOBAL_STATE["pc"] = init_pinecone(config.PINECONE_API_KEY)
    GLOBAL_STATE["embed_model"] = SentenceTransformer(config.EMBED_MODEL_NAME)
    GLOBAL_STATE["llm"] = ChatGoogleGenerativeAI(model=config.AGENT_LLM_MODEL,
                                 temperature=0.5,
                                 google_api_key=config.GOOGLE_API_KEY)
    GLOBAL_STATE["tokenizer"] = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    GLOBAL_STATE["generator"] = pipeline(
        "text-generation",
        model=config.CONTEXT_MODEL_NAME,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        truncation=True
    )
    try:
        GLOBAL_STATE["vectorizer"] = joblib.load(config.VECTORIZER_FILE_PATH)
        rag_tool = create_rag_tool(GLOBAL_STATE["pc"],
                                   GLOBAL_STATE["embed_model"],
                                   GLOBAL_STATE["vectorizer"])
        tools = [rag_tool]
        GLOBAL_STATE["agent"] = create_agent(GLOBAL_STATE["llm"],
                                             tools,
                                             system_prompt=config.AGENT_SYSTEM_PROMPT)
        print("Agent ve tüm modeller başarıyla yüklendi. Arayüz başlatılmaya hazır.")
    except FileNotFoundError:
        print(f"Uyarı: '{config.VECTORIZER_FILE_PATH}' bulunamadı.")
        print("Sistem 'devre dışı' modda başlıyor. Lütfen bir dosya yükleyerek ilk indekslemeyi başlatın.")
        GLOBAL_STATE["vectorizer"] = None
        GLOBAL_STATE["agent"] = None

except Exception as e:
    print(f"Modeller yüklenirken bir hata oluştu: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def respond(message, chat_history):
    langchain_messages = []
    for user_msg, ai_msg in chat_history:
        langchain_messages.append(("user", user_msg))
        langchain_messages.append(("assistant", ai_msg))
    langchain_messages.append(("user", message))

    print(f"\nYeni Sorgu Alındı: {message}")
    print(f"Tüm Konuşma Geçmişi: {langchain_messages}")

    result = GLOBAL_STATE["agent"].invoke({
        "messages": langchain_messages
    })

    final_answer = "Bir hata oluştu."
    if message in result and result["message"]:
        final_answer = result["message"][-1].content
        chat_history.append(("assistant", final_answer))

    print(f"Agent Cevabı: {final_answer}")
    return final_answer

def upload_and_reindex(file, chat_history):
    if file is None:
        chat_history.append((None, "Lütfen bir dosya seçin."))
        return "", chat_history

    temp_path = file.name
    file_name = os.path.basename(temp_path)
    target_path = os.path.join(config.DOC_DIR_PATH, file_name)

    try:
        shutil.copy(temp_path, target_path)
        print(f"Dosya '{target_path}' konumuna kopyalandı.")

        chat_history.append((f"({file_name} yüklendi)", "Dosya alındı. İndeksleme başlıyor... Bu işlem birkaç dakika sürebilir. Lütfen bekleyin..."))
        yield "", chat_history

    except Exception as e:
        print(f"Dosya kopyalanamadı: {e}")
        chat_history.append((None,f"Hata: Dosya kopyalanamadı: {e}"))
        yield "", chat_history
        return

    print("İndeksleme süreci ('run_rebuild.py') başlatılıyor...")
    try:
        run_rebuild(
            GLOBAL_STATE["pc"],
            GLOBAL_STATE["tokenizer"],
            GLOBAL_STATE["embed_model"],
            GLOBAL_STATE["generator"]
        )

        print("İndeksleme tamamlandır.")
        chat_history.append((None, "İndeksleme tamamlandı. Modeller hafızada güncelleniyor..."))
        yield "", chat_history
    except Exception as e:
        print(f"İndeksleme hatası: {e}")
        import traceback
        traceback.print_exc()
        chat_history.append((None, f"Hata: İndeksleme başarısız oldu.\n{e}"))
        yield "", chat_history
        return

    print("Modeller hafızada dinamik olarak yeniden yükleniyor...")
    try:
        GLOBAL_STATE["vectorizer"] = joblib.load(config.VECTORIZER_FILE_PATH)

        new_rag_tool = create_rag_tool(
            GLOBAL_STATE["pc"],
            GLOBAL_STATE["embed_model"],
            GLOBAL_STATE["vectorizer"]
        )
        GLOBAL_STATE["agent"] = create_agent(
            GLOBAL_STATE["llm"],
            [new_rag_tool],
            system_prompt=config.AGENT_SYSTEM_PROMPT
        )

        print("Hafızadaki agent başarıyla güncellendi.")
        chat_history.append((None, f"Sistem başarıyla güncellendi. Artık '{file_name}' hakkında soru sorabilirsiniz."))
        yield "", chat_history

    except Exception as e:
        print(f"Modeller yeniden yüklenirken hata oluştu: {e}")
        chat_history.append((None, f"HATA: Modeller hafızaya yeniden yüklenemedi. Lütfen uygulamayı manuel olarak yeniden başlatın. Hata: {e}"))
        yield "", chat_history


if __name__ == "__main__":
    print("Gradio Blocks arayüzü oluşturuluyor...")

    with gr.Blocks(title="RAG Agent Arayüzü", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# Belge Sorgualama Agent'ı (RAG)")
        gr.Markdown("Sistemdeki belgelere sorular sorun veya yeni belgeler ekleyin.")

        with gr.Tab("Sohbet"):
            gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(height=600, label="Sohbet Ekranı", type="messages"),
                textbox=gr.Textbox(scale=7, container=False, placeholder="Lütfen sorunuzu buraya yazınız..."),
                clear_btn="Konuşmayı Temizle"
            )

        with gr.Tab("Veritabanı Yönetimi (Admin)"):
            gr.Markdown(
                "**Yeni Belge Ekleme Adımları:**\n"
                "1. PDF dosyasını seçin ve 'Dosyayı yükle' butonuna basın.\n"
                "2. Yükleme başarılı olduktan sonra, 'Veritabanını Yeniden İndeksle' butonuna basın.\n"
                "3. İndeksleme tamamlandığında, bu uygulamayı terminalden"
            )

