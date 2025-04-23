from retriever import Retriever
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Initialize the retriever

#retriever_method = "bm25"  
retriever_method = "tfidf"
db_path = "token_semantic_chunk_mxbai_large"
retriever = Retriever(db_path=db_path, collection_name="langchain", method=retriever_method)

# Use base transformer model
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Encode text into embeddings with mean pooling
def encode(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens["attention_mask"]
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    summed = torch.sum(embeddings * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return summed / counts

def normalize(input):
    return torch.nn.functional.normalize(input, p=2, dim=1)

# Integrate retrieval with cosine similarity

def cosine_with_retrieval(query):
    # retrieve top context chunks
    contexts = retriever.retrieve(query, top_k=10)
    
    # If retriever returns a list of chunks: extract strings:
    if isinstance(contexts, list):
        contexts = [str(chunk) for chunk in contexts]

    # apply weights based 
    weights = []
    for i in range(len(contexts)):
        weight = 1.0 - (i * 0.1)
        weights.append(weight)
  

    # encode query + context
    query_embedding = normalize(encode([query]))
    context_embeddings = normalize(encode(contexts))

    # apply weights
    weighted_contexts = []
    for weights, context_embeddings in zip(weights, context_embeddings):
        weighted_context = weights * context_embeddings
        weighted_contexts.append(weighted_context)

    weighted_contexts = torch.stack(weighted_contexts)


    # compute cosine similarity
    cos_sim = cosine_similarity(query_embedding, weighted_contexts)

    # Pair each context with its similarity score
    context_scores = []
    for i, ctx in enumerate(contexts):
        context_score = (i, ctx, cos_sim[0][i])
        context_scores.append(context_score)     

    # Sort by score, descending
    context_scores.sort(key=lambda x: x[2], reverse=True)

    # Print sorted results
    for rank, (i, ctx, score) in enumerate(context_scores, 1):
        print(f"Context {rank} (Original Index {i+1})")
        print(f"Cosine Similarity: {score * 100:.2f} %")
        print(f"Context: {ctx}")


def manually_test(query):
    context_for_test = "Calibration No Pos Update Trust Revolution Counter Revolution Counter Lost Updated the value of the parameter Server Type FTP Client Communication I/O Network added to topic Communication: Enable on I/O Network Minor corrections in section Manipulator supervision K Released with RobotWare 7.7 Added the Type Move in Auto Added new action value for system parameter Action Verify Move Robot In Auto Added the new parameter Fast Device Startup Moved Type System Input Type System Output I/O System Controller Updated the Prerequisites Type System Input Updated the parameter Brake on Time Added limitation for number of instances of the types Robot Single Robot Information about Cross Connections removed from section Topic I/O System Reference added to Application Manual I/O Engineering."
    query_embedding = normalize(encode([query]))
    context_embedding = normalize(encode([context_for_test]))

    cos_sim = cosine_similarity(query_embedding, context_embedding)

    print(f"Cosine Similarity: {cos_sim[0][0] * 100:.2f} %")
    print(f"Context: {context_for_test}")


'''
# Cosine with retriever
query = "How do I move a robot linearly?"
cosine_with_retrieval(query)
'''


# Manually test the cosine similarity function
query = "How do I move a robot linearly?"
manually_test(query)




