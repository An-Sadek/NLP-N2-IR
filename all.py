import os

import numpy as np
import polars as pl
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import regex as re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from summarizer import Summarizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

Corpus = list[str]
Tokens = list[list[str]]


class NLP_Preprocessing:

	def __init__(self):
		# Regex
		self.PREFIX_PATTERN = re.compile(r"[#@/]\S+")
		self.DIGIT_PATTERN = re.compile(r"\S*\d+\S*")

		self.PUNCTUATION_TRANS = str.maketrans("", "", string.punctuation)

		self.STOPWORDS_SET = set(stopwords.words("english"))
		self.STEMMER = PorterStemmer()
		self.LEMMATIZER = WordNetLemmatizer()
		self.TAG_DICT = {
			"J": wordnet.ADJ,
			"N": wordnet.NOUN,
			"V": wordnet.VERB,
			"R": wordnet.ADV
		}

	# Xoá prefix như # @ /
	def remove_hashtag(self, corpus: Corpus) -> Tokens:
		return [
			self.PREFIX_PATTERN.sub("", sent) for sent in corpus
		]

	# Xoá các số có trong corpus
	def remove_digit(self, corpus: Tokens) -> Tokens:
		return [
			self.DIGIT_PATTERN.sub("", sent) for sent in corpus
		]

	# Token document trong corpus
	def tokenize(self, corpus: Corpus) -> Tokens:
		return [
			word_tokenize(sent) for sent in corpus
		]
	
	# Token sentence trong corpus
	def sentence_tokenize(self, corpus: Corpus) -> Tokens:
		return [
			sent_tokenize(sent) for sent in corpus
		]

	# Xoá các ký tự đặc biệt
	def remove_punctuation(self, tokens: Tokens) -> Tokens:
		return [
			[
				word.translate(self.PUNCTUATION_TRANS) for word in sublist if word.translate(self.PUNCTUATION_TRANS)
			] 
			for sublist in tokens
		]

	# Đổi lại chữ thường
	def lowercase(self, tokens: Tokens) -> Tokens:
		return [
			[
				word.lower() for word in sublist
			] 
			for sublist in tokens
		]

	# Xoá stopwords, các từ ko mang quá nhiều giá trị
	def remove_stopwords(self, tokens: Tokens, custom_stopwords: set[str] = None) -> Tokens:
		stopwords = self.STOPWORDS_SET if custom_stopwords is None else set(custom_stopwords)

		return [
			[
				word for word in sublist if word not in stopwords
			] 
			for sublist in tokens
		]

	# Stem
	def stem(self, tokens: Tokens) -> Tokens:
		return [
			[
				self.STEMMER.stem(word) for word in sublist
			] 
			for sublist in tokens
		]

	# Lấy loại từ
	def get_wordnet_pos(self, tagged_tokens):
		return [
			(
				word, self.TAG_DICT.get(tag[0].upper(), wordnet.NOUN)
			) 
			for word, tag in tagged_tokens
		]

	# Lemm
	def lemmatize(self, tokens: Tokens) -> Tokens:
		return [
			[
				self.LEMMATIZER.lemmatize(word, pos) for word, pos in self.get_wordnet_pos(pos_tag(sublist))
			] 
			for sublist in tokens
		]

	# Tiền xử lý toàn bộ các phương thức trên
	def preprocess(self, corpus: Corpus) -> Tokens:
		corpus = self.remove_hashtag(corpus)
		corpus = self.remove_digit(corpus)
		tokens = self.tokenize(corpus)
		tokens = self.lowercase(tokens)
		tokens = self.remove_punctuation(tokens)
		tokens = self.remove_stopwords(tokens)
		tokens = self.stem(tokens)
		#tokens = self.lemmatize(tokens)
		return tokens

	# Trả về corpus thay vì token
	def get_corpus(self, tokens: Tokens) -> Corpus:
		corpus = [
			" ".join(sub_list) for sub_list in tokens
		]

		return corpus
	

class SteamDataset:

	def __init__(self, path, n_rows=2500, seed=42):
		assert os.path.exists(path), f"File không tồn tại: {path}"
		assert n_rows % 2 == 0, "n_rows phải chia hết cho 2"

		self.path = path
		data = pl.read_csv(path)
		data = data[:, ["review_text", "review_votes"]]
		data = data.drop_nulls()

		self.n_rows = n_rows
		half = n_rows // 2 # Lấy phân nửa

		# Shufle để đảm bảo tính ngẫu nhiên không thiên về game nào
		shuffled_data = data.sample(fraction=1, shuffle=True, seed=seed)
		votes_0 = shuffled_data.filter(pl.col("review_votes") == 0).head(half)
		votes_1 = shuffled_data.filter(pl.col("review_votes") == 1).head(half)
		small_data = pl.concat([votes_0, votes_1]) 
		# Không cần shuffle lần 2 vì khi chia dữ liệu có thể shuffle tiếp

		tools = NLP_Preprocessing()
		self.tokens = tools.preprocess(small_data["review_text"])
		self.corpus = tools.get_corpus(self.tokens)
		self.label = small_data["review_votes"]

	def __len__(self):
		return self.n_rows

	def __getitem__(self, idx):
		return self.tokens[idx], self.corpus[idx], self.label[idx]
	
	def get_cfs_mtx(self, model, train_size=0.8, seed=42):
		# Use corpus (preprocessed text) instead of tokens for vectorization
		X_train, X_test, y_train, y_test = train_test_split(self.corpus, self.label, shuffle=True, 
														random_state=seed, 
														train_size=train_size)
		
		# Create and fit TF-IDF vectorizer
		vectorizer = TfidfVectorizer()
		X_train_vec = vectorizer.fit_transform(X_train).toarray()
		X_test_vec = vectorizer.transform(X_test).toarray()

		# Train and predict
		model.fit(X_train_vec, y_train)
		y_pred = model.predict(X_test_vec)
		return confusion_matrix(y_test, y_pred)

	def plot_cfs_mtx(self, mtx):
		sns.heatmap(mtx/np.sum(mtx), annot=True)
		plt.show()


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


if __name__ == "__main__":
	"""
	# NLP preprocessing
	corpus = ["This is 1st document", "This is 2nd document"]
	tools = NLP_Preprocessing()
	
	tokens = tools.preprocess(corpus)
	pre_corpus = tools.get_corpus(tokens)

	print(tokens)
	print(pre_corpus)

	# Dataset preprocessing
	dataset = SteamDataset("./dataset.csv", n_rows=100)
	print(dataset[0])
	print(dataset[-1])"""

	# Simple Preprocessing
	tools = NLP_Preprocessing()
	corpus = ["This is 1st document", "This is 2nd document"]
	print(tools.remove_stopwords(corpus))

	# Text Summarization
	ts = TextSummarization(path="./text.txt")

	text = ts.text
	print("\n\nLSA")
	print(ts.LSA(text))

	print("\n\nLexRank")
	print(ts.LexRank(text))

	print("\n\nTextRank")
	print(ts.TextRank(text))

	print("\n\nBert")
	print(ts.Bert(text))

	# ML-based sentiment analysis
	dataset = SteamDataset("./dataset.csv", n_rows=100)
	
	lr = LogisticRegression()
	lr_cfs_mtx = dataset.get_cfs_mtx(lr)
	dataset.plot_cfs_mtx(lr_cfs_mtx)

	nb = GaussianNB()
	nb_cfs_mtx = dataset.get_cfs_mtx(nb)
	dataset.plot_cfs_mtx(nb_cfs_mtx)
