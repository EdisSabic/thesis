from retriever import Retriever
import ollama
import time as time
from tqdm import tqdm

class ABBot:
    """
    Args:
        retrievr (Retriever): An instance of the Retriever class for document retrieval.
        llm (str): The name of the LM to use e.g "Gemma3:1b".
        embedding (str): The name of the embedding model to use e.g "mxbai-embed-large".
    """
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.chat_history = [
            {"role": "system", "content": "You are an AI assistant that provides answers based on the given context."}
        ]

    def run(self):
        print("Welcome to the ABBot.\nEnter /help for available commands.")
        while True:
            query = input("Enter your question: ")
            start_time = time.time()
            if query.lower() == "/exit":
                break

            elif query.lower() == "/reset":
                self.chat_history = [
                    {"role": "system", "content": "You are an AI assistant that provides answers based on the given context."}
                ]
                print("Chat history reset.")
                continue

            elif query.lower() == "/help":
                print(f"""Available commands:
                      - /exit: Exit the chat
                      - /reset: Reset the chat history
                      - /help: Show this help message
                      """)
                continue

            elif query.lower() == "/benchmark":
                iterations = 10
                total_time = 0
                benchmark_query = input("Enter question to benchmark: ")
                print("Running benchmark...")
                iteration_results = []

                for i in tqdm(range(iterations)):
                    self.chat_history = [
                    {"role": "system", "content": "You are an AI assistant that provides answers based on the given context."}
                ]
                    start_time = time.time()
                    context = self.retriever.retrieve(benchmark_query, top_k=3)
                    user_message = f"Context: {context}\n\nQuestion: {benchmark_query}"
                    self.chat_history.append({"role": "user", "content": user_message})
                    response = ollama.chat(self.llm, self.chat_history)
                    answer = response["message"]["content"]
                    self.chat_history.append({"role": "assistant", "content": answer})
                    print(f"\nABBot: {answer}\n")
                    total_time += time.time() - start_time
                    iteration_results.append(time.time() - start_time)

                print(f"Average execution time: {total_time / 10:.2f} seconds.")
                print(f"Total execution time: {total_time:.2f} seconds.")

                for i in range(0, 10):
                    print(f"Iteration {i+1}: {iteration_results[i]:.2f} seconds")



            else:
                context = self.retriever.retrieve(query, top_k=3)

                user_message = f"Context: {context}\n\nQuestion: {query}"
                self.chat_history.append({"role": "user", "content": user_message})

                response = ollama.chat(self.llm, self.chat_history)
                answer = response["message"]["content"]

                self.chat_history.append({"role": "assistant", "content": answer})

                print(f"\nABBot: {answer}\n")
                print(f"Execution time: {time.time() - start_time:.2f} seconds")



persist_path = "C:/Users/SEEDSAB/Desktop/Thesis/chroma_semantic_chunk_mxbai_large"
retriever_tfidf = Retriever(db_path=persist_path, collection_name="langchain", method="tfidf")
retriever_bm25 = Retriever(db_path=persist_path, collection_name="langchain", method="bm25")

ABBot_tfidf = ABBot(retriever_tfidf, llm="Gemma3:1b")
ABBot_bm25 = ABBot(retriever_bm25, llm="llama3.2:1b")


#ABBot_bm25.run()
ABBot_tfidf.run()