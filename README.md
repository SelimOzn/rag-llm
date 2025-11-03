# Gelişmiş RAG Agent Projesi (Hybrid Search & Contextualization)

Bu proje, yüklenen PDF belgeleri üzerinden soruları yanıtlamak için tasarlanmış gelişmiş bir Retrieval-Augmented Generation (RAG) sistemidir. Standart RAG yaklaşımlarının ötesine geçerek, arama isabetliliğini artırmak için **Hibrit Arama (Hybrid Search)**, **Anlamsal Parçalama (Semantic Chunking)** ve **LLM ile Otomatik Bağlamsallaştırma (Contextualization)** gibi modern teknikleri kullanır.

Sistem, bir LangChain Agent'ı ve Gradio arayüzü ile paketlenmiştir, bu sayede kullanıcılar hem sohbet edebilir hem de dinamik olarak yeni belgeler yükleyip indeksi güncelleyebilir.

## 🚀 Temel Özellikler

* **Hibrit Arama (Hybrid Search):** Hem anahtar kelime tabanlı (Sparse, TF-IDF) hem de anlamsal (Dense, SBERT) aramayı birleştirerek her iki dünyanın da avantajlarını kullanır. Sonuçlar normalleştirilir ve ağırlıklı bir skorla birleştirilir.
* **Anlamsal Parçalama (Semantic Chunking):** Metinleri sabit boyutlu parçalara bölmek yerine, anlamsal olarak ilişkili cümleleri (cümleler arası kosinüs benzerliğine göre) bir arada tutan bir `hybrid_chunker` kullanır.
* **LLM ile Bağlamsallaştırma:** İndekslemeden *önce*, her bir metin parçasının (chunk) belge içindeki yerini daha iyi açıklaması için küçük bir LLM (`meta-llama/Llama-3.2-1B-Instruct`) kullanarak özet bir bağlam (context) üretilir ve bu, parçanın başına eklenir. Bu, arama sırasında alaka düzeyini önemli ölçüde artırır.
* **Akıllı Agent:** Sorguları işlemek ve `DocumentHybridSearch` aracını akıllıca kullanmak için `ChatGoogleGenerativeAI` (Gemini) modeli ile güçlendirilmiş bir LangChain agent'ı içerir.
* **Dinamik İndeksleme:** Gradio arayüzü üzerinden yeni PDF'ler yüklendiğinde tüm veri işleme (ingestion) pijplini (`run_rebuild`) otomatik olarak tetiklenir ve agent hafızada güncellenir.

## ⚙️ Mimari ve Veri Akışı

Proje, iki ana aşamadan oluşur: **Veri İşleme (Ingestion)** ve **Sorgulama (Inference)**.

### 1. Veri İşleme (Ingestion) Pijplini
Yeni bir PDF yüklendiğinde (`app.py` -> `upload_and_reindex` -> `run_rebuild`):

1.  **PDF Ayrıştırma:** PDF, `PyMuPDF` (fitz) kullanılarak başlıklarına (font kalınlığı ve boşluklara göre) ayrıştırılır ve `(başlık, içerik)` çiftleri olarak kaydedilir.
2.  **Anlamsal Parçalama:** Her bölümün içeriği, `hybrid_chunker` ile anlamsal olarak tutarlı parçalara (chunks) bölünür.
3.  **Bağlamsallaştırma:** Her parça, `CONTEXT_MODEL_NAME` (`Llama-3.2-1B`) modeline gönderilerek bir "bağlam" özeti üretilir ve bu özet parçanın başına eklenir (`context + chunk_text`).
4.  **Vektörleştirme (Dense):** Bağlamsallaştırılmış parçalar `EMBED_MODEL_NAME` (`all-MiniLM-L6-v2`) ile gömme (embedding) vektörlerine dönüştürülür.
5.  **Vektörleştirme (Sparse):** Tüm parçalar üzerinde bir `TfidfVectorizer` eğitilir (`vectorizer.joblib` olarak kaydedilir) ve sparse vektörler oluşturulur.
6.  **İndeksleme:** Dense ve Sparse vektörler, iki ayrı Pinecone sunucusuz (serverless) indeksine (`rag-dense` ve `rag-sparse`) yüklenir.

### 2. Sorgulama (Inference) Akışı

1.  **Girdi:** Kullanıcı, Gradio arayüzünden bir soru sorar.
2.  **Agent:** LangChain agent'ı, soruyu analiz eder ve `DocumentHybridSearch` aracını kullanmaya karar verir.
3.  **Hibrit Arama:**
    * Sorgu, hem dense (embedding) hem de sparse (TF-IDF) vektörlere dönüştürülür.
    * Her iki Pinecone indeksinde de `top_k` arama yapılır.
    * Sonuçların skorları normalleştirilir (`normalize_scores`) ve `alpha` değerine göre birleştirilerek yeniden sıralanır.
4.  **Yanıt Üretme:** En iyi `top_k` sonuç (JSON formatında) agent'a geri gönderilir.
5.  **Sonuç:** Agent, bu arama sonuçlarını (context) kullanarak doğal dilde bir yanıt oluşturur ve kullanıcıya sunar.

## 🛠️ Teknoloji Stack'i

* **LLM & Agent:** LangChain, Google Gemini (via `langchain-google-genai`), Transformers
* **Vektör Veritabanı:** Pinecone (Serverless)
* **Embedding & Vektörleştirme:** SentenceTransformers, Scikit-learn (TfidfVectorizer)
* **Arayüz (UI):** Gradio
* **Veri İşleme:** PyMuPDF (fitz), NLTK, Joblib
* **Altyapı (Opsiyonel):** Docker (Pinecone local test servisleri için)

## 🏁 Kurulum ve Çalıştırma

### 1. Ön Gereksinimler

* Python 3.12
* Pinecone Hesabı (API Key için)
* Google AI Studio Hesabı (Gemini API Key için)
* Hugging Face Hesabı (Llama modelleri için Token)

### 2. Kurulum

1.  **Projeyi klonlayın:**
    ```bash
    git clone [https://github.com/kullanici-adiniz/rag-llm.git](https://github.com/kullanici-adiniz/rag-llm.git)
    cd rag-llm
    ```

2.  **Sanal ortam oluşturun ve aktifleştirin:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (Windows için: venv\Scripts\activate)
    ```

3.  **Bağımlılıkları yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **NLTK Verisini İndirin:**
    `pipeline/chunking.py` dosyası NLTK'nın `punkt` modülünü kullanıyor. İndirmek için:
    ```bash
    python -m nltk.downloader punkt_tab
    ```

5.  **`.env` Dosyasını Oluşturun:**
    Proje ana dizininde `.env` adında bir dosya oluşturun ve `utils/config.py` dosyasına göre aşağıdaki değişkenleri doldurun:

    ```env
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"

    # Llama 3.1 ve 3.2 modelleri 'gated' (erişim kısıtlamalı) modellerdir.
    # Bu modelleri kullanabilmek için Hugging Face Hub token'ınıza ihtiyacınız olabilir.
    HUGGINGFACE_HUB_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"
    ```

### 3. Çalıştırma

Uygulamayı başlatmak için `app.py` dosyasını çalıştırın:

```bash
python app.py
```

## 📂 Proje Yapısı

Projenin ana dizin yapısı ve önemli dosyaların açıklamaları aşağıdadır. `docs`, `saves` gibi klasörler `config.py` içinde tanımlanmıştır ve uygulama çalıştırıldığında (`app.py`) otomatik olarak oluşturulur.
```
rag-llm/
│
├── app.py                      # Gradio arayüzünü başlatan ve agent'ı yükleyen ana uygulama dosyası
├── docker-compose.yml          # (Opsiyonel) Pinecone local test servislerini başlatmak için
├── README.md                   # Bu döküman
│
├── pipeline/
│ ├── init.py                   # Pipeline modüllerini import edilebilir hale getirir
│ ├── chunking.py               # Anlamsal parçalama (semantic chunking) mantığını içerir
│ └── contextualize.py          # LLM ile parçalara bağlam ekleme mantığını içerir
│
├── utils/
│ ├── init.py                   # Yardımcı fonksiyonları import edilebilir hale getirir
│ ├── config.py                 # Tüm konfigürasyonları, API anahtarlarını ve dosya yollarını yönetir
│ ├── index_conf.py             # Pinecone indekslerini oluşturma, silme ve sorgulama fonksiyonları
│ ├── index_manager.py          # Tüm veri işleme (ingestion) pipeline’ını (run_rebuild) yönetir
│ ├── io_utils.py               # JSONL dosyalarına yazma gibi I/O işlemleri
│ ├── pdf_utils.py              # PDF dosyalarını ayrıştıran (parsing) fonksiyonlar
│ └── rag_core.py               # Hibrit arama (hybrid_search) ve skor normalleştirme mantığı
│
├── docs/                       # (Dinamik) Yüklenecek PDF'lerin konulduğu klasör
├── processed_docs/             # (Dinamik) İşlemi tamamlanan PDF'lerin taşındığı klasör
├── saves/                      # (Dinamik) İşleme sırasında üretilen ara dosyaların (chunks.jsonl, docs.jsonl vb.) kaydedildiği yer
└── sparse_vectorizer/          # (Dinamik) Eğitilmiş TF-IDF modelinin (vectorizer.joblib) kaydedildiği yer
```
