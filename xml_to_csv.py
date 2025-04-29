import xml.etree.ElementTree as ET
import re
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import csv

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

if __name__ == "__main__":
    xml_file_3HAC032104 = "C:\\Skola\\Exjobb\\Git_code\\data\\3HAC032104 OM RobotStudio_a631_en\\index.xml"
    xml_file_3HAC065038 = "C:\\Skola\\Exjobb\\Git_code\\data\\3HAC065038 TRM RAPID RW 7_a631_en\\index.xml"
    xml_file_3HAC065040 = "C:\\Skola\\Exjobb\\Git_code\\data\\3HAC065040 RAPID Overview RW 7_a631_en\\index.xml"
    xml_file_3HAC065041 = "C:\\Skola\\Exjobb\\Git_code\\data\\3HAC065041 TRM System parameters RW 7_a631_en\\index.xml"
    xml_files = [
        xml_file_3HAC032104,
        xml_file_3HAC065038,
        xml_file_3HAC065040,
        xml_file_3HAC065041
    ]

    embed_model = OllamaEmbeddings(model = "mxbai-embed-large")

    for file_path in xml_files:
        raw_text = extract_text_from_xml(file_path)
        cleaned_text = clean_text(raw_text)
        chunks = semantic_chunking(cleaned_text)
        csv_file_name = f"{file_path.split('\\')[-2]}.csv"
        with open(csv_file_name, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Chunk"])
            for chunk in chunks:
                writer.writerow([chunk])
        
        print(f"CSV file created: {csv_file_name}")
        print(f"Total chunks: {len(chunks)}")