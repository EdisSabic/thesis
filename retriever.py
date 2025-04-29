import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

class Retriever:
    def __init__(self, db_path, collection_name, method):
        """
        Args: 
            db_path (str): Path to the ChromaDB database.
            collection_name (str): Name of the collection in the database.
            method (str): Retrieval method to use ('tfidf' or 'bm25').
        """
        self.method = method.lower()

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

        self.documents = []
        self.ids = []
        self._load_documents()

        if self.method == "tfidf":
            self._init_tfidf()
        elif self.method == "bm25":
            self._init_bm25()
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'tfidf' or 'bm25'.")


    def _load_documents(self):
        results = self.collection.get()
        self.documents = results['documents']
        self.ids = results['ids']
        #print(f"Loaded {len(self.documents)} documents from the ChromaDB.")

    def _init_tfidf(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    
    def _init_bm25(self):
        self.tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def retrieve(self, query, top_k=5):
        if self.method == "tfidf":
            return self._retrieve_tfidf(query, top_k)
        else:
            return self._retrieve_bm25(query, top_k)
    
    def _retrieve_tfidf(self, query, top_k):
        query_vector = self.vectorizer.transform([query])
        scores = self.tfidf_matrix.dot(query_vector.T).toarray().flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.ids[i], self.documents[i], scores[i]) for i in top_indices]
    
    def _retrieve_bm25(self, query, top_k):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], self.documents[i], scores[i]) for i in top_indices]

if __name__ == "__main__":
    db_path = "C:\\Users\\SEEDSAB\\Desktop\\ThesisABB\\thesis\\token_semantic_chunk_mxbai_large"
    collection_name = "langchain"
    query = "In basic RAPID programming, what are the restrictions of Multitasking RAPID?"


    retriever = Retriever(db_path, collection_name, method="tfidf")
    results = retriever.retrieve(query, top_k=3)
    
    for i, (doc_id, doc, score) in enumerate(results):
        print(f"[{i+1}] ID: {doc_id} | Score: {score:.4f}\n{doc}\n")