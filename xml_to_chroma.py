import xml.etree.ElementTree as ET
import re
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

def extract_text_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    text_parts = []

    for elem in root.iter():
        if elem.text:
            text_parts.append(elem.text.strip())

    return "\n".join(text_parts)


def clean_text(text):
    # Normalize quotes and dashes
    text = text.replace('“', '"').replace('”', '"').replace('’', "'")
    text = text.replace('–', '-').replace('—', '-')

    # Remove any remaining XML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove any remaining special characters
    re.sub(r'[^\w\s.,!?\'\":;\-\(\)]', '', text)

    # Remove headers and footers
    text = re.sub(r'^\s*Header.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*Footer.*?$', '', text, flags=re.MULTILINE)

    # Remove repeated substrings
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def chunk_text(text, chunk_size=1600, overlap=100):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    return chunks

def semantic_chunking(text, chunk_size=1600, overlap=100):
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )

    chunks = splitter.split_text(text)
    return chunks

def count_tokens(text):
    return len(text.split()) * 1.3

def token_semantic_chunk(text, max_tokens=512):
    """
    Splits input text into semantically coherent chunks based on sentence boundaries 
    and estimated token limits, with awareness of document structure cues.

    Tokenizes the input text into individual sentences using NLTK's 
    sentence tokenizer. It then groups sentences into chunks without exceeding a specified 
    maximum token count (`max_tokens`). Additionally, the chunking logic is aware of 
    semantic boundaries — such as headings or document structure markers like "Section" 
    or "Table" — which will force a new chunk to begin even if the token limit 
    hasn't been reached yet.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        # Semantic break conditions
        if re.match(r'^(Section|Table)\b', sentence):
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


if __name__ == "__main__":
    xml_file_3HAC032104 = "C:/Users/SEEDSAB/Desktop/files/3HAC032104 OM RobotStudio_a631_en/index.xml"
    xml_file_3HAC065038 = "C:/Users/SEEDSAB/Desktop/files/3HAC065038 TRM RAPID RW 7_a631_en/index.xml"
    xml_file_3HAC065040 = "C:/Users/SEEDSAB/Desktop/files/3HAC065040 RAPID Overview RW 7_a631_en/index.xml"
    xml_file_3HAC065041 = "C:/Users/SEEDSAB/Desktop/files/3HAC065041 TRM System parameters RW 7_a631_en/index.xml"
    xml_files = [
        xml_file_3HAC032104,
        xml_file_3HAC065038,
        xml_file_3HAC065040,
        xml_file_3HAC065041
    ]

    embed_model = OllamaEmbeddings(model = "mxbai-embed-large")

    for file_path in tqdm(xml_files):
        raw_text = extract_text_from_xml(file_path)
        cleaned_text = clean_text(raw_text)
        chunks = token_semantic_chunk(cleaned_text, max_tokens=512)

        print(f"File: {file_path.split('\\)[-2]')}")
        print(f"Total chunks: {len(chunks)}")
        
        vector_store = Chroma(embedding_function=embed_model, persist_directory="./token_semantic_chunk_mxbai_large")
        vector_store.add_texts(chunks)