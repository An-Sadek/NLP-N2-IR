import nltk
import regex as re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin

# Downloads data
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("wordnet")
nltk.download("punkt")
nltk.download('punkt_tab')

# Regex
PREFIX_PATTERN = re.compile(r"[#@/]\S+")
DIGIT_PATTERN = re.compile(r"\S*\d+\S*")

PUNCTUATION_TRANS = str.maketrans("", "", string.punctuation)

# Một số biến, clas sử dụng
STOPWORDS_SET = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
TAG_DICT = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV
}

# Xoá prefix như # @ /
def remove_hashtag(corpus: list[str]) -> list[str]:
    return [PREFIX_PATTERN.sub("", sent) for sent in corpus]

# Xoá các số có trong corpus
def remove_digit(corpus: list[str]) -> list[str]:
    return [DIGIT_PATTERN.sub("", sent) for sent in corpus]

# Token document trong corpus
def tokenize(corpus: list[str]) -> list[list[str]]:
    return [word_tokenize(sent) for sent in corpus]

# Xoá các ký tự đặc biệt
def remove_punctuation(tokens: list[list[str]]) -> list[list[str]]:
    return [[word.translate(PUNCTUATION_TRANS) for word in sublist if word.translate(PUNCTUATION_TRANS)] for sublist in tokens]

# Đổi lại chữ thường
def lowercase(tokens: list[list[str]]) -> list[list[str]]:
    return [[word.lower() for word in sublist] for sublist in tokens]

# Xoá stopwords, các từ ko mang quá nhiều giá trị
def remove_stopwords(tokens: list[list[str]]) -> list[list[str]]:
    return [[word for word in sublist if word not in STOPWORDS_SET] for sublist in tokens]

# Stem
def stem(tokens: list[list[str]]) -> list[list[str]]:
    return [[STEMMER.stem(word) for word in sublist] for sublist in tokens]

# Lấy loại từ
def get_wordnet_pos(tagged_tokens):
    return [(word, TAG_DICT.get(tag[0].upper(), wordnet.NOUN)) for word, tag in tagged_tokens]

# Lemm
def lemmatize(tokens: list[list[str]]) -> list[list[str]]:
    return [[LEMMATIZER.lemmatize(word, pos) for word, pos in get_wordnet_pos(pos_tag(sublist))] for sublist in tokens]

# Tiền xử lý toàn bộ các phương thức trên
def preprocess(corpus: list[str]) -> list[list[str]]:
    corpus = remove_hashtag(corpus)
    corpus = remove_digit(corpus)
    tokens = tokenize(corpus)
    tokens = lowercase(tokens)
    tokens = remove_punctuation(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    #tokens = lemmatize(tokens)
    return tokens

# Trả về corpus thay vì token
def get_corpus_list(tokens: list[list[str]])->list[str]:
    corpus = [
        " ".join(sub_list) for sub_list in tokens
    ]

    return corpus

if __name__ == "__main__":
    corpus = [
        "I love Fear & Hunger: Termina. The atmosphere is great!",
        "I hate fast-paced animes. @Asriel #Marinathebest 1234 /1"
    ]

    print("\nOriginal Corpus:")
    print(corpus)

    print("\nProcessed Corpus:")
    print(preprocess(corpus))

    print("\nJoin list")
    print(get_corpus_list(preprocess(corpus)))
