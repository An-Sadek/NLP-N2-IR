import nltk
import regex as re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("punkt")

# Compile regex patterns for efficiency
PREFIX_PATTERN = re.compile(r"[#@/]\S+")
DIGIT_PATTERN = re.compile(r"\S*\d+\S*")

PUNCTUATION_TRANS = str.maketrans("", "", string.punctuation)

# Define NLP tools
STOPWORDS_SET = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
TAG_DICT = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV
}

def remove_hashtag(corpus: list[str]) -> list[str]:
    """Removes hashtags, mentions, and slashes."""
    return [PREFIX_PATTERN.sub("", sent) for sent in corpus]

def remove_digit(corpus: list[str]) -> list[str]:
    """Removes words containing digits."""
    return [DIGIT_PATTERN.sub("", sent) for sent in corpus]

def tokenize(corpus: list[str]) -> list[list[str]]:
    """Tokenizes text into words."""
    return [word_tokenize(sent) for sent in corpus]

def remove_punctuation(tokens: list[list[str]]) -> list[list[str]]:
    """Removes punctuation using str.translate for efficiency."""
    return [[word.translate(PUNCTUATION_TRANS) for word in sublist if word.translate(PUNCTUATION_TRANS)] for sublist in tokens]

def lowercase(tokens: list[list[str]]) -> list[list[str]]:
    """Converts all words to lowercase."""
    return [[word.lower() for word in sublist] for sublist in tokens]

def remove_stopwords(tokens: list[list[str]]) -> list[list[str]]:
    """Removes stopwords using a set for faster lookup."""
    return [[word for word in sublist if word not in STOPWORDS_SET] for sublist in tokens]

def stem(tokens: list[list[str]]) -> list[list[str]]:
    """Applies stemming using PorterStemmer."""
    return [[STEMMER.stem(word) for word in sublist] for sublist in tokens]

def get_wordnet_pos(tagged_tokens):
    """Maps POS tags to WordNet tags."""
    return [(word, TAG_DICT.get(tag[0].upper(), wordnet.NOUN)) for word, tag in tagged_tokens]

def lemmatize(tokens: list[list[str]]) -> list[list[str]]:
    """Lemmatizes words with their respective POS tags."""
    return [[LEMMATIZER.lemmatize(word, pos) for word, pos in get_wordnet_pos(pos_tag(sublist))] for sublist in tokens]

def preprocess(corpus: list[str]) -> list[list[str]]:
    """Full preprocessing pipeline."""
    corpus = remove_hashtag(corpus)
    corpus = remove_digit(corpus)
    tokens = tokenize(corpus)
    tokens = lowercase(tokens)
    tokens = remove_punctuation(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    tokens = lemmatize(tokens)
    return tokens

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
