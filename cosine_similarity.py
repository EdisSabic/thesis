from retriever import Retriever
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch


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
        if score >= 0.7500:
            print(f"Context {rank} (Original Index {i+1})")
            print(f"Cosine Similarity: {score * 100:.2f} %")
            print(f"Context: {ctx}")


def manually_test(query):
    context_for_test = "('af4bdc14-2624-4810-849d-2a69770a2c46', SEMISTATIC task A SEMISTATIC task gets restarted from the beginning whenever the power is turned on. A SEMISTATIC task will also initiate the restart sequence, reload modules specified in the system parameters if the module file is newer than the loaded module. Station Viewer It can playback a station in 3D without RobotStudio. It packages the station file together with the files needed to view the station in 3D. It can also play recorded simulations. Simit SIMIT is a simulation platform from Siemens for virtual commissioning of factory automation. Symbol A signal is identified by this name in SIMIT. T Tool A tool is an object that can be mounted directly or indirectly on the robot turning. Tooldata A tool is represented with a variable of the data type tooldata Tool Centre Point (TCP) Refers to the point in relation to which robot's positioning is defined. It is the center point of the tool coordinate system that defines the position and orientation of the tool. TCP has its zero position at the center point of the tool. The tool center point also constitutes the origin of the tool coordinate system. Robot system can handle a number of TCP definitions, but only one can be active. Task A task is an activity or piece of work. RobotStudio tasks are either Normal, Static or Semistatic. Task frame Represents the origin of the robot controller world coordinate system in RobotStudio. Track motion A mechanism consisting of a linear axis with a carriage on which the robot is mounted. The track motion is used to give the robot improved reachability while working with large work pieces. Target Target signifies the position to which the robot is programmed to move. It is a RobotStudio object that contains the position and orientation of the point that the robot must reach. Position data is used to define the position in the move instructions to which the robot and additional axes will move. As the robot is able to achieve the same position in several different ways, the axis configuration is also specified. Target object contains values that shows position of the robot, orientation of the tool, axis configuration of the robot and position of the additional logical axes.', 0.06953206448380607)"
    query_embedding = normalize(encode([query]))
    context_embedding = normalize(encode([context_for_test]))

    cos_sim = cosine_similarity(query_embedding, context_embedding)

    print(f"Cosine Similarity: {cos_sim[0][0] * 100:.2f} %")
    print(f"Context: {context_for_test}")



# Cosine with retriever
query = "How do I move a robot linearly?"
cosine_with_retrieval(query)


'''
# Manually test the cosine similarity function
query = "How do I move a robot linearly?"
manually_test(query)
'''



