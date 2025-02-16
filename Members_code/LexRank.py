import numpy as np
import networkx as nx
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

# Tải dữ liệu NLTK nếu chưa có
nltk.download('punkt_tab')

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    return text.lower()

# Hàm đọc văn bản từ file .txt
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

# Hàm tính LexRank
def lexrank_summarization(text, num_sentences=3):
    # 1. Tách câu
    sentences = sent_tokenize(text)
    
    # 2. Vector hóa các câu bằng TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # 3. Tính ma trận cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 4. Xây dựng đồ thị và áp dụng PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # 5. Chọn các câu quan trọng nhất
    ranked_sentences = sorted(((scores[i], sent) for i, sent in enumerate(sentences)), reverse=True)
    summary = " ".join([sent for _, sent in ranked_sentences[:num_sentences]])

    return summary

# Đường dẫn tới file .txt (thay bằng file của bạn)
file_path = "./text.txt"

# Đọc nội dung file
text = read_text_file(file_path)

# Gọi hàm tóm tắt
summary = lexrank_summarization(text, num_sentences=3)
print("===== BẢN TÓM TẮT =====")
print(summary)
