import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model (any sentence embedding model works)
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Utility: encode and normalize vectors
def encode_text(text_list):
    tokens = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**tokens)
    embeddings = model_output.last_hidden_state.mean(dim=1)  # mean pooling
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

# Sample inputs
questions = ["What is a hash table?", "Explain merge sort."]
contexts = [
    "A hash table is a data structure that maps keys to values for efficient lookup.",
    "Merge sort is a divide-and-conquer sorting algorithm."
]
weights = [0.8, 0.6]  # Example context relevance weights

# Step 1: Encode all texts
question_embs = encode_text(questions)
context_embs = encode_text(contexts)

# Step 2: Apply weights to context embeddings
weighted_context_embs = torch.stack([w * e for w, e in zip(weights, context_embs)])

# Step 3: Compute similarity matrix
similarity_matrix = cosine_similarity(question_embs, weighted_context_embs)

# Step 4: Show results
for i, q in enumerate(questions):
    for j, c in enumerate(contexts):
        print(f"Similarity(Q{i} <-> C{j}): {similarity_matrix[i][j]:.4f}")
