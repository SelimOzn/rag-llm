import os
from dotenv import load_dotenv

load_dotenv()
#API anahtarları
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Adları
TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CONTEXT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
AGENT_LLM_MODEL = "gemini-2.0-flash"

# Dosya/Klasör yolları
DOC_DIR_PATH = "docs"
PROCESSED_DOCS_DIR = "processed_docs"
SAVES_DIR = "saves"
CHUNK_SAVE_PATH = os.path.join(SAVES_DIR, "chunks.jsonl")
CONTEXTED_SAVE_PATH = os.path.join(SAVES_DIR, "contexts.jsonl")
TITLE_SAVE_PATH = os.path.join(SAVES_DIR, "titles.jsonl")
DOC_SAVE_PATH = os.path.join(SAVES_DIR, "docs.jsonl")
VECTORIZER_DIR_PATH = "./sparse_vectorizer/"
VECTORIZER_FILE_PATH = os.path.join(VECTORIZER_DIR_PATH, "vectorizer.joblib")

# Index ayarları
DENSE_INDEX_NAME = "rag-dense"
SPARSE_INDEX_NAME = "rag-sparse"

# İşleme parametreleri
MAX_TOKENS = 200
SIMILARITY_THRESHOLD = 0.5
SPARSE_BATCH_SIZE = 100

# Prompt Şablonları
CONTEXT_PROMPT_TEMPLATE = """
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
AGENT_SYSTEM_PROMPT = """
    Sen, belge tabanlı soruları yanıtlayan bir asistansın.
    Kullanıcı bir soru sorduğunda, görevin cevabı DocumentHybridSearch aracını 
    kullanarak bulmaktır. 
    Sana verilen JSON string'i (arama sonuçlarını) kullanarak kullanıcıya 
    doğal dilde bir cevap oluştur.
    Cevabı bilmediğini varsayma, HER ZAMAN önce arama aracını kullan.
    """
