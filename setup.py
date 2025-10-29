import os

from sentence_transformers import SentenceTransformer

from utils import config, init_pinecone, create_dense_index, create_sparse_index


def initialize_project():
    print("Proje altyapısı kuruluyor...")

    os.makedirs(config.DOC_DIR_PATH, exist_ok=True)
    os.makedirs(config.PROCESSED_DOCS_DIR, exist_ok=True)
    os.makedirs(config.SAVES_DIR, exist_ok=True)
    os.makedirs(config.VECTORIZER_DIR_PATH, exist_ok=True)
    print(f"Gerekli klasörler ('{config.DOC_DIR_PATH}', '{config.PROCESSED_DOCS_DIR}' vb.) hazır.")

    try:
        pc = init_pinecone(config.PINECONE_API_KEY)
    except Exception as e:
        print(f"Hata: Pinecone bağlantısı kurulamadı. API anahtarınızı kontrol edin. {e}")
        return

    try:
        embed_model = SentenceTransformer(config.EMBED_MODEL_NAME)
        dense_vector_dim = embed_model.get_sentence_embedding_dimension()
        del embed_model
    except Exception as e:
        print(f"Hata: Embedding modeli ('{config.EMBED_MODEL_NAME}') yüklenemedi. {e}")
        return

    create_dense_index(pc, config.DENSE_INDEX_NAME, dense_vector_dim)
    create_sparse_index(pc, config.SPARSE_INDEX_NAME)

    print("\n[✓] Kurulum tamamlandı. Pinecone indeksleri hazır.")
    print("Artık 'build_index.py' (ilk veri yüklemesi için) veya 'app.py' (uygulamayı başlatmak için) çalıştırabilirsiniz.")


if __name__ == "__main__":
    initialize_project()



