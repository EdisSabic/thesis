import xml.etree.ElementTree as ET

def extract_text_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Get all text content from the XML tree
    text_parts = []

    for elem in root.iter():
        if elem.text:
            text_parts.append(elem.text.strip())

    return "\n".join(text_parts)


def analyze_text(text):
    paragraphs = [p for p in text.split('\n') if p.strip() != '']
    words = text.split()
    total_words = len(words)
    total_length = sum(len(word.strip('.,!?()[]{}"\'')) for word in words)
    avg_word_length = total_length / total_words if total_words else 0
    paragraph_count = len(paragraphs)

    return total_words, avg_word_length, paragraph_count

if __name__ == "__main__":
    xml_file_3HAC032104 = "C:/Users/SEEDSAB\Desktop/files/3HAC032104 OM RobotStudio_a631_en/index.xml"
    xml_file_3HAC065038 = "C:/Users/SEEDSAB\Desktop/files/3HAC065038 TRM RAPID RW 7_a631_en/index.xml"
    xml_file_3HAC065040 = "C:/Users/SEEDSAB\Desktop/files/3HAC065040 RAPID Overview RW 7_a631_en/index.xml"
    xml_file_3HAC065041 = "C:/Users/SEEDSAB\Desktop/files/3HAC065041 TRM System parameters RW 7_a631_en/index.xml"
    xml_files = [
        xml_file_3HAC032104,
        xml_file_3HAC065038,
        xml_file_3HAC065040,
        xml_file_3HAC065041
    ]
    xml_strings = [
        "3HAC032104 OM RobotStudio_a631_en",
        "3HAC065038 TRM RAPID RW 7_a631_en",
        "3HAC065040 RAPID Overview RW 7_a631_en",
        "3HAC065041 TRM System parameters RW 7_a631_en"
    ]
    for i in xml_files:
        text = extract_text_from_xml(i)
        total_words, avg_length, paragraphs = analyze_text(text)
        print("")
        print("File name", xml_strings[xml_files.index(i)])
        print(f"Word count: {total_words}")
        print(f"Average word length: {avg_length:.2f}")
        print(f"Paragraph count: {paragraphs}")


""" 
Results from running the script:

File name 3HAC032104 OM RobotStudio_a631_en
Word count: 76879
Average word length: 5.27
Paragraph count: 15751

File name 3HAC065038 TRM RAPID RW 7_a631_en
Word count: 262642
Average word length: 5.31
Paragraph count: 68833

File name 3HAC065040 RAPID Overview RW 7_a631_en
Word count: 29624
Average word length: 5.09
Paragraph count: 5040

File name 3HAC065041 TRM System parameters RW 7_a631_en
Word count: 58351
Average word length: 5.32
Paragraph count: 17616
"""