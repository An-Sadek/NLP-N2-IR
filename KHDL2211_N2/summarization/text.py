import os
import sys
sys.path.append("../preprocessing")
from preprocessing.nlp_tools import NLP_Preprocessing

import numpy as np
import networkx as nx

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

from summarizer import Summarizer

Corpus = list[str]
Tokens = list[list[str]]



class TextSummarization:
	
	def __init__(self, path: str=None, text: str=None):
		assert (path is None) ^ (text is None), "Chỉ sử dụng path hoặc text"

		self.text = None
		if not path is None:
			assert os.path.exists(path), f"Đường dẫn không tồn tại: {path}"

			file = open(path, "r+")
			self.text = file.read()
		else:
			self.text = text

		self.tools = NLP_Preprocessing()
		
	def LSA(self, text: str, n_sentences:int = 3):
		sentences = sent_tokenize(text)
		assert n_sentences <= len(sentences), "Số câu tóm tắt không được lớn hơn số câu trong văn bản"

		documents = [' '.join(sentences)]

		# Tính toán ma trận TF-IDF
		vectorizer = TfidfVectorizer(stop_words='english')
		X = vectorizer.fit_transform(documents)

		"""
		Giảm chiều dữ liệu. 
		`n_components=2` cho biết số lượng các thành phần chính (components) mà chúng ta muốn giữ lại từ ma trận TF-IDF.
		`algorithm='randomized'` sử dụng phương pháp ngẫu nhiên để tính toán SVD, giảm độ phức tạp
		`n_iter=100` xác định số lần lặp tối đa. `random_state=122` thiết lập giá trị ngẫu nhiên cố định để đảm bảo tính tái lập kết quả.
		"""
		svd_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=100, random_state=122)
		lsa = make_pipeline(svd_model, Normalizer(copy=False))
		X_lsa = lsa.fit_transform(X)

		# Lấy từ từ ma trận TF-IDF
		terms = vectorizer.get_feature_names_out()

		# Lấy các thành phần (components) của mô hình SVD sau khi đã huấn luyện, 
		# mỗi thành phần tương ứng với một "concept" (khái niệm).
		concepts = svd_model.components_
		summary = [] # Tạo mảng rỗng để lưu trữ các khái niệm tóm tắt.
		for i, concept in enumerate(concepts):
			sorted_terms = [terms[j] for j in concept.argsort()[:-5 - 1:-1]]
			summary.append(f"Concept {i}: {' '.join(sorted_terms)}")

		# Lọc các câu quan trọng
		important_sentences = []
		for sentence in sentences: # Duyệt qua từng câu 
			if any(term in sentence for term in sorted_terms): # Kiểm tra xem câu có chứa bất kỳ từ nào trong danh sách từ quan trọn
				important_sentences.append(sentence)
		
		# Chọn các câu quan trọng nhất
		important_sentences = important_sentences[:n_sentences] 

		summary_text = ' '.join(important_sentences)

		return summary_text
	
	def LexRank(self, text, n_sentences:int = 3):
		# 1. Tách câu
		sentences = sent_tokenize(text)

		# 2. Vector hóa các câu bằng TF-IDF
		vectorizer = TfidfVectorizer()
		tfidf_matrix = vectorizer.fit_transform(sentences)

		# 3. Tính ma trận cosine similarity
		similarity_matrix = cosine_similarity(tfidf_matrix)

		# 4. Xây dựng đồ thị và áp dụng PageRank
		nx_graph = nx.from_numpy_array(similarity_matrix)

		# 4. Xây dựng đồ thị và áp dụng PageRank
		graph = nx.from_numpy_array(similarity_matrix)
		scores = nx.pagerank(graph)

		# 5. Chọn các câu quan trọng nhất
		ranked_sentences = sorted(((scores[i], sent) for i, sent in enumerate(sentences)), reverse=True)
		summary = " ".join([sent for _, sent in ranked_sentences[:n_sentences]])

		return summary

	# TextRank được tối ưu
	def TextRank(self, text, n_sentences:int = 3, dim=100, n_iter=50):
		# Lemmatizer
		lemmatizer = WordNetLemmatizer()
		
		# Process text once
		sentences = sent_tokenize(text.lower())
		
		# More efficient stopwords check
		custom_stopwords = set(stopwords.words('english')) - {"not", "without"}
		
		# Process sentences more efficiently
		processed_sentences = []
		for s in sentences:
			words = [lemmatizer.lemmatize(word) for word in word_tokenize(s) 
					if word.isalpha() and word not in custom_stopwords]
			processed_sentences.append(" ".join(words))

		# Load embeddings more efficiently - only load common words
		word_embeddings = {}
		common_words = set()
		for sent in processed_sentences:
			common_words.update(sent.split())
			
		with open("./glove.6B.100d.txt", encoding="utf-8") as f:
			for line in f:
				values = line.split()
				word = values[0]
				if word in common_words:  # Only load needed embeddings
					coefs = np.asarray(values[1:], dtype='float32')
					word_embeddings[word] = coefs
		
		# Chuyển đổi câu thành vector bằng cách tính trung bình vector của các từ trong câu.
		sentence_vectors = []
		for sentence in processed_sentences:
			if not sentence:
				sentence_vectors.append(np.zeros(dim))
				continue
			words = sentence.split()
			word_vectors = [word_embeddings.get(w, np.zeros(dim)) for w in words]
			sentence_vectors.append(np.mean(word_vectors, axis=0))
		
		# Tính ma trận cosine similarity
		sentence_vectors = np.array(sentence_vectors)
		sim_mat = cosine_similarity(sentence_vectors)
		np.fill_diagonal(sim_mat, 0)
		
		nx_graph = nx.from_numpy_array(sim_mat)
		scores = nx.pagerank(nx_graph, max_iter=50)  # Limit max iterations
		
		# Sắp xếp các câu theo độ quan trọng
		ranked_sentences = [(scores[i], s) for i, s in enumerate(sentences)]
		ranked_sentences.sort(reverse=True)
		summary = "\n".join(sent for _, sent in ranked_sentences[:n_sentences])
		
		return summary

	def Bert(self, text, n_sentences:int = 3):
		bert_model = Summarizer()
		summary = bert_model(text, num_sentences=n_sentences)
		return summary
