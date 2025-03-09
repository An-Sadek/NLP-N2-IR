import os
import sys 
sys.path.append(r"preprocessing")
from nlp_tools import NLP_Preprocessing


import numpy as np
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import (
	CountVectorizer,
	TfidfVectorizer
)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


class SteamDataset:

	def __init__(self, path, n_rows=100, seed=42, lemm=False):
		"""
		Steam dataset có khoảng 4 500 000 dòng
		Để lemm = True để thực hiện lemm
		"""
		assert os.path.exists(path), f"File không tồn tại: {path}"
		assert n_rows % 2 == 0, "n_rows phải chia hết cho 2"

		print("===Đang tiến hành lấy dữ liệu===")
		self.path = path
		data = pl.read_csv(path)
		data = data[:, ["review_text", "review_votes"]]
		data = data.drop_nulls()
		print("===Lấy hoàn tất===\n")

		self.n_rows = n_rows
		half = n_rows // 2 # Lấy phân nửa

		# Shufle để đảm bảo tính ngẫu nhiên không thiên về game nào
		print("===Trộn và lấy dữ liệu nhỏ hơn===")
		shuffled_data = data.sample(fraction=1, shuffle=True, seed=seed)
		votes_0 = shuffled_data.filter(pl.col("review_votes") == 0).head(half)
		votes_1 = shuffled_data.filter(pl.col("review_votes") == 1).head(half)
		small_data = pl.concat([votes_0, votes_1]) 
		print("===Trộn hoàn tất===\n")
		# Không cần shuffle lần 2 vì khi chia dữ liệu có thể shuffle tiếp

		print("===Tiền xử lý văn bản===")
		tools = NLP_Preprocessing()
		self.documents = small_data["review_text"]
		self.tokens = tools.preprocess(small_data["review_text"], lemm)
		self.corpus = tools.get_corpus(self.tokens)
		self.label = small_data["review_votes"]
		print("===Tiền xử lý hoàn tất===\n")

	def __len__(self):
		return self.n_rows

	def __getitem__(self, idx):
		return self.documents, self.tokens[idx], self.corpus[idx], self.label[idx]
	
	def get_cfs_mtx(self, model, train_size=0.7, seed=42, vectorizer=None):
		X_train, X_test, y_train, y_test = train_test_split(
			self.corpus, self.label, 
			stratify=self.label,
			shuffle=True, 
			random_state=seed, 
			train_size=train_size
		)
		
		X_train_vec = vectorizer.fit_transform(X_train).toarray()
		X_test_vec = vectorizer.transform(X_test).toarray()

		# Train and predict
		model.fit(X_train_vec, y_train)
		y_pred = model.predict(X_test_vec)
		return confusion_matrix(y_test, y_pred)

	def plot_cfs_mtx(self, mtx):
		sns.heatmap(mtx/np.sum(mtx), annot=True)
		plt.show()


if __name__ == "__main__":
	# Load data
	data = SteamDataset("../dataset.csv", 100, 42)
	print("Số lượng dòng đã lấy: ", len(data))
	print("Văn bản gốc: \n", data[0][0])
	print("Văn bản được tiền xử lý: \n", data[0][1])
	print("Token: \n", data[0][2])
	print("Đánh giá: \n", data[0][3])

	# Bags of words
	bow_vectorizer = CountVectorizer()

	## Naive bayes
	nb1 = GaussianNB()
	nb1_cfs_mtx = data.get_cfs_mtx(nb1, vectorizer=bow_vectorizer)
	data.plot_cfs_mtx(nb1_cfs_mtx)

	## Logistic Regression
	lr1 = LogisticRegression()
	lr1_cfs_mtx = data.get_cfs_mtx(lr1, vectorizer=bow_vectorizer)
	lr1_cfs_mtx = np.where(lr1_cfs_mtx >= 0.5, 1, 0)
	data.plot_cfs_mtx(lr1_cfs_mtx)

	# TF-IDF
	tfidf_vectorizer = TfidfVectorizer()

	## Naive bayes
	nb2 = GaussianNB()
	nb2_cfs_mtx = data.get_cfs_mtx(nb2, vectorizer=tfidf_vectorizer)
	data.plot_cfs_mtx(nb2_cfs_mtx)

	## Logistic Regression
	lr2 = LogisticRegression()
	lr2_cfs_mtx = data.get_cfs_mtx(lr2, vectorizer=tfidf_vectorizer)
	lr2_cfs_mtx = np.where(lr2_cfs_mtx >= 0.5, 1, 0)
	data.plot_cfs_mtx(lr2_cfs_mtx)



