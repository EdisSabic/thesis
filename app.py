import streamlit as st
from retriever import Retriever
import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class ABBot:
    def __init__(self, retriever, llm, embedding, db_path):
        self.retriever = retriever
        self.llm = llm
        self.embedding = embedding
        self.db_path = db_path
        self.chat_history = [
            {"role": "system", "content": "You are an AI assistant that provides answers based on the given context."}
        ]

    def get_response(self, query):
        context = self.retriever.retrieve(query, top_k=3)
        user_message = f"Context: {context}\n\nQuestion: {query}"
        self.chat_history.append({"role": "user", "content": user_message})

        response = ollama.chat(self.llm, self.chat_history)
        answer = response["message"]["content"]

        self.chat_history.append({"role": "assistant", "content": answer})
        return answer

# Initialize retrievers and ABBot instances
persist_path = "C:/Users/SEEDSAB/Desktop/Thesis/chroma_semantic_chunk_mxbai_large"
retriever_tfidf = Retriever(db_path=persist_path, collection_name="langchain", method="tfidf")
retriever_bm25 = Retriever(db_path=persist_path, collection_name="langchain", method="bm25")
ABBot_tfidf = ABBot(retriever_tfidf, llm="Gemma3:1b", embedding="mxbai-embed-large", db_path=persist_path)
ABBot_bm25 = ABBot(retriever_bm25, llm="Gemma3:1b", embedding="mxbai-embed-large", db_path=persist_path)

# Streamlit app layout
st.title("ABBot Chatbot")

# Select retriever method
retriever_method = st.selectbox("Select Retriever Method", ["TF-IDF", "BM25"])
if retriever_method == "TF-IDF":
    chatbot = ABBot_tfidf
else:
    chatbot = ABBot_bm25

# User input
user_input = st.text_input("You: ", "")
if user_input:
    if user_input.lower() == "/reset":
        chatbot.chat_history = [
            {"role": "system", "content": "You are an AI assistant that provides answers based on the given context."}
        ]
        st.write("Chat history reset.")
    elif user_input.lower() == "/help":
        st.write("""Available commands:
        - /exit: Exit the chat
        - /reset: Reset the chat history
        - /help: Show this help message""")
    else:
        response = chatbot.get_response(user_input)
        st.write(f"Bot: {response}")
