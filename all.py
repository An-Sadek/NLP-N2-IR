import os

import numpy as np
import polars as pl

import nltk
import regex as re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

Corpus = list[str]
Tokens = list[list[str]]

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
    def remove_stopwords(self, tokens: Tokens) -> Tokens:
        return [
            [
                word for word in sublist if word not in self.STOPWORDS_SET
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
        tokens = self.lemmatize(tokens)
        return tokens

    # Trả về corpus thay vì token
    def get_corpus(self, tokens: Tokens) -> Corpus:
        corpus = [
            " ".join(sub_list) for sub_list in tokens
        ]

        return corpus
    

class RuleSentimentAnalysis:

    def __init__(self):
        pass

if __name__ == "__main__":
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
    print(dataset[-1])