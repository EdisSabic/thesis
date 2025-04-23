# pip install transformers torch scikit-learn

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# Use a base model that works locally with PyTorch
model_name = "bert-base-uncased"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Encode text into embeddings with mean pooling
def encode(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    # Mean pooling
    attention_mask = tokens["attention_mask"]
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    summed = torch.sum(embeddings * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / counts
    return mean_pooled

# Normalize the vectors (L2 norm)
def normalize(vecs):
    return torch.nn.functional.normalize(vecs, p=2, dim=1)

# Input: questions, contexts, weights
questions = ["What is a hash table?", "Explain merge sort."]
contexts = [
    "A hash table is a data structure that maps keys to values for efficient lookup.",
    "Merge sort is a divide-and-conquer sorting algorithm."
]
weights = [0.8, 0.6]  # Context weights

# Encode
question_embeddings = normalize(encode(questions))
context_embeddings = normalize(encode(contexts))

# Apply weights to context vectors
weighted_contexts = torch.stack([weight * vec for weight, vec in zip(weights, context_embeddings)])

# Compute cosine similarity
cos_sim = cosine_similarity(question_embeddings, weighted_contexts)

# Display results
print("\nWeighted Cosine Similarity Matrix (Questions × Contexts):\n")
for i, q in enumerate(questions):
    for j, c in enumerate(contexts):
        print(f"Q{i+1} \"{q}\" <-> C{j+1}: {cos_sim[i][j]:.4f}")
