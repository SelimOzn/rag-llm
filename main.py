import json
import re
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
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

if __name__ == '__main__':
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
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
    DENSE_INDEX_NAME = "rag-dense"
    SPARSE_INDEX_NAME = "rag-sparse"
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
    print(dense_vector_dim)
    # context_tokenizer = AutoTokenizer.from_pretrained(context_model_name, token=token)
    #
    # # Eksik pad token düzeltmesi
    # if context_tokenizer.pad_token is None:
    #     context_tokenizer.pad_token = context_tokenizer.eos_token
    #context_model = AutoModelForCausalLM.from_pretrained(context_model_name, token=token)
    generator = pipeline("text-generation",
                         model=context_model_name,
                         max_new_tokens=100,
                         temperature=0.7,
                         top_p=0.9,
                         truncation=True)

    bm25_vectorizer = TfidfVectorizer()

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

    all_chunks = []
    metadata = []
    with open(contexted_save_path, "r") as f:
        for line in f:
            contexted_chunk = json.loads(line)
            contexted_text = contexted_chunk["chunk"]
            tokens = re.findall(r'\w+', contexted_text.lower())
            all_chunks.append(tokens)
            metadata.append(contexted_chunk)

    bm25 = BM25Okapi(all_chunks)
    create_sparse_index(pc, SPARSE_INDEX_NAME)
    sparse_vectors = []
    for i, tokens in enumerate(tqdm(all_chunks, desc="Creating sparse vectors")):
        token_weights = {}
        for token in set(tokens):
            tf = tokens.count(token)
            idf = bm25.idf.get(token, 0)
            token_weights[token] = tf * idf  # BM25 skorunun temel hali

        # Pinecone formatına çevir
        indices = list(range(len(token_weights)))  # gerçek token ID'leri tokenizer'dan alınabilir
        values = list(token_weights.values())

        sparse_vectors.append({
            "id": f"chunk_{i}",
            "sparse_values": {
                "indices": indices,
                "values": values
            },
            "metadata": metadata[i]
        })
    sparse_index_upsert(pc, SPARSE_INDEX_NAME, sparse_vectors)







