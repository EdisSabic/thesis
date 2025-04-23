from retriever import Retriever
import ollama
import time as time
from tqdm import tqdm
import sys
import os
from questions import robostudio_questions, rw_rapid_questions, overview_rapid_questions, rw_system_parameters_questions, all_questions

class ABBot:
    """
    Args:
        retriever (Retriever): An instance of the Retriever class for document retrieval.
        llm (str): The name of the LLM to use from local Ollama models.
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.chat_history = [
            {"role": "system", "content": "You are an AI assistant that provides answers based on the given context."}
        ]
         
    def _chat_(self, query):
            context = self.retriever.retrieve(query, top_k=3)
            print(context)
            user_message = f"Context: {context}\n\nQuestion: {query}"
            self.chat_history.append({"role": "user", "content": user_message})

            response = ollama.chat(self.llm, self.chat_history)
            answer = response["message"]["content"]

            self.chat_history.append({"role": "assistant", "content": answer})
            print(f"\nABBot: {answer}\n")

    def _benchmark_(self):
                total_time = 0
                iteration_results = []
                question_list_types = [all_questions, robostudio_questions, rw_rapid_questions, overview_rapid_questions, rw_system_parameters_questions]

                choice = input("""Enter a number 1-5 to choose a question set:
                                  \n1. All Questions\n2. Robostudio Questions\n3. RW RAPID Questions\n4. Overview RAPID Questions\n5. RW System Parameters Questions\nYour choice: """)
                
                questions = question_list_types[int(choice) - 1]
                selected_list = question_list_types[int(choice) - 1]
                list_name = [name for name, value in globals().items() if value is selected_list][0]
                iterations = int(input("Enter the number of iterations to run benchmark on: "))

                print(f"\n\nRunning benchmark for {iterations} iterations on question set: {list_name}")
                print("-" * 100)
                for question in tqdm(questions):
                    for i in range(0, iterations):
                        print("*" * 100)
                        print(f"Question: {question}")
                        self._reset_chat_history_()

                        start_time = time.time()

                        self._chat_(question)

                        iteration_time = time.time() - start_time

                        total_time += iteration_time

                        iteration_results.append(iteration_time)
                        print(f"Question: {question}\nIteration {i+1}: {iteration_results[i]:.2f} seconds\n\n")
                        
                    iteration_results.clear()
                    print("-" * 100)

                questions_answered = len(questions) * int(iterations)
                print(f"Benchmark completed -- Retriever: {self.retriever.method} used with LLM: {self.llm}\n\n")
                print(f"Average response time: {total_time / questions_answered:.2f} seconds.")
                print(f"Total execution time on {questions_answered} questions: {total_time:.2f} seconds.")
                print(f"Question set used: {list_name}")
                print(f"Number of iterations: {iterations}")
                
    def _reset_chat_history_(self):
                self.chat_history = [
                    {"role": "system", "content": "You are an AI assistant that provides answers based on the given context."}
                    ]

    def chat(self):
        print(f"Current LLM: {self.llm}")
        print(f"Current Retriever: {self.retriever.method}")

        print("\nWelcome to the ABBot.\nEnter /help for available commands.")

        while True:
            query = input("Enter your question: ")

            start_time = time.time()

            if query.lower() == "/exit":
                print("Exiting the chat.")
                break

            elif query.lower() == "/reset":
                self._reset_chat_history_()
                print("Chat history reset.")

            elif query.lower() == "/help":

                print(f"""Available commands:
                      - /exit: Exit the chat
                      - /reset: Reset the chat history
                      - /help: Show this help message
                      - /benchmark: Run a benchmark test
                      - /cls or /clear: Clear the output""")

            elif query.lower() == "/benchmark":
                 self._benchmark_()
            
            elif query.lower() == "/clear" or query.lower() == "/cls":
                 os.system('cls' if os.name=='nt' else 'clear')

            else:
                self._chat_(query)
                print("-" * 100)
                print(f"Execution time: {time.time() - start_time:.2f} seconds\n\n")


if len(sys.argv) != 3:
    print("Usage: python ABBot.py <retriever_method> <llm_choice>")
    print("Current supported retrievers: tfidf, bm25")
    print("Exiting ABBot, please try again.")
    exit()

db_path = "token_semantic_chunk_mxbai_large"

retriever_method, llm_choice = sys.argv[1], sys.argv[2]
retriever = Retriever(db_path=db_path, collection_name="langchain", method=retriever_method)

abbot = ABBot(retriever=retriever, llm=llm_choice)
abbot.chat()