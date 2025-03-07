import os
import sys 
sys.path.append("../preprocessing")
from preprocessing.nlp_tools import NLP_Preprocessing

import numpy as np
import polars as pl
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import regex as re
import string
from nltk.tokenize import word_tokenize, sent_tokenize


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix

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

