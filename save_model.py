from sentence_transformers import SentenceTransformer

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

model_path = "./models/embedding_model"
model.save_pretrained(model_path)