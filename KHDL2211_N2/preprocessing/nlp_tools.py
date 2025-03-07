import os
import sys

import regex as re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

import spacy
nlp = spacy.load("en_core_web_sm")


Corpus = list[str]
Tokens = list[list[str]]


class NLP_Preprocessing:
	
	def __init__(self):
		# Regex
		self.VALID_PATTERN = re.compile(r"[^a-zA-z0-9\s]")
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

		self.voca = set(nlp.vocab.strings)
	
	# Xoá các chuỗi không hợp lệ
	def remove_invalid(self, corpus: Corpus) -> Corpus:
		return [
			self.VALID_PATTERN.sub("", sent) for sent in corpus
		]

	# Xoá các số có trong corpus
	def remove_digit(self, corpus: Corpus) -> Tokens:
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
	
	# Kiểm tra từ điển
	def check_vocabulary(self, tokens: Tokens)->Tokens:
		return[
			[
				word for word in sublist if word in self.voca
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
		corpus = self.remove_invalid(corpus)
		corpus = self.remove_digit(corpus)
		tokens = self.tokenize(corpus)
		tokens = self.lowercase(tokens)
		tokens = self.check_vocabulary(tokens)
		tokens = self.remove_stopwords(tokens)
		tokens = self.remove_punctuation(tokens)
		tokens = self.stem(tokens)
		#tokens = self.lemmatize(tokens)
		return tokens

	# Trả về corpus thay vì token
	def get_corpus(self, tokens: Tokens) -> Corpus:
		corpus = [
			" ".join(sub_list) for sub_list in tokens
		]

		return corpus
	
if __name__ == "__main__":
	pass