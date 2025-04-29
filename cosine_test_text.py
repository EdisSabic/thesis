import os
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

answers_path = "C:\\Users\\SEEDSAB\\Desktop\\ThesisABB\\thesis\\answers"
azure_path = "C:\\Users\\SEEDSAB\\Desktop\\ThesisABB\\thesis\\azure_answers\\Answers_Azure_Bot.docx"

# Read text from docx file
def read_docx(answer_path):
    doc = Document(answer_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text.strip()

# Read documents
ground_truth = read_docx(azure_path)

documents = []
filenames = []

for filename in os.listdir(answers_path):
    if filename.endswith(".docx"):
        filepath = os.path.join(answers_path, filename)
        text = read_docx(filepath)
        documents.append(text)
        filenames.append(filename)

# Removing stopwords
stop_words = set(stopwords.words('english'))

# Clean the text
def preprocess_text(text):
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Preprocess documents
ground_truth = preprocess_text(ground_truth)
documents = [preprocess_text(doc) for doc in documents]

# Vectorize the text with TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([ground_truth] + documents)

# Calculate cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

for filename, similarity in zip(filenames, cosine_similarities):
    print(f"Ground truth / {filename}: {similarity * 100:.2f}%")
