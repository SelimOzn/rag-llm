from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
import os
from utils.pdf_utils import save_jsonl
import json
import functools

def hybrid_chunker(
    text,
    emb_model,
    tokenizer,
    similarity_threshold,
    max_tokens,
):

    # Tokenize text into sentences
    sentences = sent_tokenize(text)

    # Encode sentences
    embeddings = emb_model.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk = [sentences[0]]
    current_len = len(tokenizer.encode(sentences[0], add_special_tokens=False))

    for i in range(1, len(sentences)):
        sent_tokens = len(tokenizer.encode(sentences[i], add_special_tokens=False))
        similarity = util.pytorch_cos_sim(embeddings[i - 1], embeddings[i]).item()

        # Check both semantic and token budget
        if (
            similarity > similarity_threshold
            and current_len + sent_tokens <= max_tokens
        ):
            current_chunk.append(sentences[i])
            current_len += sent_tokens

        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_len = sent_tokens

    # Add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_title(entry, tokenizer, model, similarity_threshold, max_tokens):
    title = entry["title"]
    content = entry["content"]
    doc_id = entry["doc_id"]

    chunks = hybrid_chunker(content,
                            emb_model=model,
                            tokenizer=tokenizer,
                            similarity_threshold=similarity_threshold,
                            max_tokens=max_tokens)

    return [{"doc_id":doc_id, "title":title, "chunk_id":i, "chunk":chunk}
            for i,chunk in enumerate(chunks)]


def concurrent_chunker(entries,
                       save_file,
                       emb_model,
                       tokenizer,
                       similarity_threshold=0.85,
                       max_tokens=200):



    results = []
    chunker_fuck = functools.partial(
        process_title,
        tokenizer=tokenizer,
        model=emb_model,
        similarity_threshold=similarity_threshold,
        max_tokens=max_tokens
    )

    with ProcessPoolExecutor() as executor:
        for out in executor.map(chunker_fuck, entries):
            results.extend(out)

    save_jsonl(results, save_file)
    return results

if __name__ == "__main__":
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
    title_file_path = "../saves/titles.jsonl"
    chunk_save_path = "../saves/chunks.jsonl"
    MAX_TOKENS = 200
    SIMILARITY_THRESHOLD = 0.5
    entries = []
    with open(title_file_path, "r") as f:
        for line in f:
            entries.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token)
    embed_model = SentenceTransformer(embed_model_name)

    chunks = concurrent_chunker(entries,
                                chunk_save_path,
                                emb_model=embed_model,
                                tokenizer=tokenizer,
                                max_tokens=MAX_TOKENS,
                                similarity_threshold=SIMILARITY_THRESHOLD)

    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}:\n{c}\n")
