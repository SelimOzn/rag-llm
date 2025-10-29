from utils import dense_index_query,sparse_index_query

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