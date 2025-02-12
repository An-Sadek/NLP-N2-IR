import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import regex as re

from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.base import BaseEstimator, TransformerMixin


prefix_pattern = r"[#@/]\S+"
digit_pattern = r"\S*\d+\S*"
punctuation = string.punctuation

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
tag_dict = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV
}


def remove_hashtag(corpus: list[str]) -> list[str]:
    no_prefix = [
        re.sub(prefix_pattern, "", sent) for sent in corpus
    ]

    return no_prefix

def remove_digit(corpus: list[str])->list[str]:
    no_digit = [
        re.sub(digit_pattern, "", sent) for sent in corpus
    ]

    return no_digit

def tokenization(corpus: list[str])->list[list[str]]:
    tokens = [
        word_tokenize(word) for word in corpus
    ]

    return tokens

def remove_punctuation(tokens: list[list[str]]):
    no_punc = [
        [
            word for word in sub_list if not word in punctuation
        ] for sub_list in tokens
    ]

    return no_punc

def lower_word(tokens: list[list[str]]):
    lower = [
        [
            word.lower() for word in sub_list
        ] for sub_list in tokens
    ]

    return lower

def remove_stopwords(tokens: list[list[str]]):
    no_stopwords = [
        [
            word for word in sub_list if word not in stopwords.words('english')
        ] for sub_list in tokens
    ]

    return no_stopwords

def stemming(tokens: list[list[str]]):
    stem_words = [
        [
            stemmer.stem(word) for word in sub_list
        ] for sub_list in tokens
    ]

    return stem_words

def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        return tag_dict.get(tag, wordnet.NOUN)

def lemming(tokens: list[list[str]]):
    lem_words = [
        [
            lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in sub_list
        ] for sub_list in tokens
    ]

    return lem_words

def preprocess_all(corpus: list[str]):
    prefix_removal = remove_hashtag(corpus)
    digit_removal = remove_digit(prefix_removal)
    tokens = tokenization(digit_removal)
    lower_words = lower_word(tokens)
    punctuation_removal = remove_punctuation(lower_words)
    stopword_removal = remove_stopwords(punctuation_removal)
    stem_word = stemming(stopword_removal)
    lem_word = lemming(stem_word)

    return lem_word


if __name__ == "__main__":
    corpus = [
        "I love Fear & Hunger: Termina. The atmosphere is great.",
        "I hate fast-paced animes. @Asriel #Marinathebest 1234 /1"
    ]

    no_hashtag = remove_hashtag(corpus)
    print(no_hashtag)

    no_digit = remove_digit(no_hashtag)
    print(no_digit)

    tokens = tokenization(no_digit)
    print(tokens)

    no_punc = remove_punctuation(tokens)
    print(no_punc)

    lower = lower_word(no_punc)
    print(lower)

    no_stopwords = remove_stopwords(lower)
    print(no_stopwords)
    
    stem = stemming(no_stopwords)
    print(stem)

    lem = lemming(stem)
    print(lem)
    
    print("\n\nTest pipeline")
    result = preprocess_all(corpus)
    print(result)