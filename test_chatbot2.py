import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

client = chromadb.Client()

persist_directory = "C:/Users/SEEDSAB/Desktop/Thesis/chroma_semantic_chunk_mxbai_large"

embedding = OllamaEmbeddings(model="mxbai-embed-large")

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

raw_docs = vectorstore.get(include=["documents", "metadatas"])
documents = [Document(page_content=doc, metadata=meta if isinstance(meta, dict) else {}) for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])]

bm25_retriever = BM25Retriever.from_documents(documents=documents)

similarity_search_retriever = vectorstore.as_retriever()

# Ensemble the retrievers
ensemble_retriever = EnsembleRetriever(retrievers=[similarity_search_retriever, bm25_retriever], weights=[0.5, 0.5])

llm = OllamaLLM(model="gemma3:1b", temperature=0.5)

rag_chain = ConversationalRetrievalChain.from_llm(llm, ensemble_retriever)

query = "from what document can i find where i get instructions for how to move a robot linearly?"

response = rag_chain.invoke({"question": query, "chat_history": []})
print("Question:", query)
print("Answer:", response['answer'])
