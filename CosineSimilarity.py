import os
import chardet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    result = chardet.detect(rawdata)
    return result['encoding']

def read_file(file_path, encoding=None):
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        return f.read()

def preprocess_text(text):
    return text.lower()

def get_file_contents(folder_path):
    file_contents = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            encoding = detect_encoding(file_path)
            content = read_file(file_path, encoding)
            file_contents[file_name] = preprocess_text(content)
    return file_contents

def calculate_cosine_similarity(file_contents, target_file_content):
    vectorizer = TfidfVectorizer()
    files = list(file_contents.values()) + [target_file_content]
    tfidf_matrix = vectorizer.fit_transform(files)
    
    target_vector = tfidf_matrix[-1]
    cosine_similarities = cosine_similarity(tfidf_matrix[:-1], target_vector.reshape(1, -1)).flatten()
    
    return cosine_similarities

def main(folder_path, target_file_path):
    target_file_encoding = detect_encoding(target_file_path)
    target_file_content = preprocess_text(read_file(target_file_path, target_file_encoding))
    
    file_contents = get_file_contents(folder_path)
    file_names = list(file_contents.keys())
    cosine_similarities = calculate_cosine_similarity(file_contents, target_file_content)
    
    sorted_indices = np.argsort(cosine_similarities)[::-1]
    sorted_files = [(file_names[idx], cosine_similarities[idx]) for idx in sorted_indices]
    
    for file_name, similarity in sorted_files:
        print(f"Cosine Similarity for {file_name} is: {similarity:.4f}")

if __name__ == "__main__":
    folder_path = "dataset_folder/"
    target_file_path = "dataset_folder/query.txt"
    main(folder_path, target_file_path)
