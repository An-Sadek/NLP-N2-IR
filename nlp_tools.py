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


class Preprocessing:

    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self.hashtag_uname_pattern = r"(?:#|@)\w+\s*"
        self.digit_pattern = r"\S*\d+\S*"
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

    def preprocess_all(self):
        self.remove_hashtag_username()
        self.tokenization()
        self.lower_word()
        self.remove_punctuation()
        self.remove_hashtag_username()
        self.remove_number()
        self.stemming()
        self.lemmatization()

    def remove_hashtag_username(self):
        self.corpus = [re.sub(self.hashtag_uname_pattern, '', text).strip() for text in self.corpus]

    def tokenization(self):
        self.corpus = [word_tokenize(sentence) for sentence in self.corpus]

    def lower_word(self):
        self.corpus =  [
                [
                    word.lower() for word in sub_list
                ] for sub_list in self.corpus
            ]

    def remove_stopwords(self):
        self.corpus = [
                [
                    word for word in sub_list if word not in stopwords.words('english')
                ] for sub_list in self.corpus
            ]
        
    def remove_punctuation(self):
        self.corpus = [
            [
                word for word in sub_list if word not in string.punctuation
            ] for sub_list in self.corpus
        ]
        
    def remove_number(self):
        self.corpus = [
            [
                re.sub(self.digit_pattern, '', text).strip() for text in sub_list
            ] for sub_list in self.corpus
        ]

        self.corpus = [
            [
                word for word in sub_list if word
            ] for sub_list in self.corpus
        ]

    def stemming(self):
        self.corpus = [
            [
                self.stemmer.stem(word) for word in sub_list
            ] for sub_list in self.corpus
        ]

    def lemmatization(self):
        self.corpus = [
            [
                self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in sub_list
            ] for sub_list in self.corpus
        ]

    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        return self.tag_dict.get(tag, wordnet.NOUN)
    
if __name__ == "__main__":
    corpus = [
        "The mechanism of pattern recognition in the brain is \
        little known, and it seems to be almost impossible to \
        reveal it only by conventional physiological experiments.", 
        
        "So, we take a slightly different approach to this. #A @1234 5k Android 4.1"]
    pre_tool = Preprocessing(corpus)
    print("Original")
    print(pre_tool.corpus)

    print("\nRemove hashtags & username")
    pre_tool.remove_hashtag_username()
    print(pre_tool.corpus)

    print("\nTokenization")
    pre_tool.tokenization()
    print(pre_tool.corpus)

    print("\nLower")
    pre_tool.lower_word()
    print(pre_tool.corpus)

    print("\nRemove punctuation")
    pre_tool.remove_punctuation()
    print(pre_tool.corpus)

    print("\nRemove stopwords")
    pre_tool.remove_stopwords()
    print(pre_tool.corpus)

    print("\nRemove num")
    pre_tool.remove_number()
    print(pre_tool.corpus)

    print("\nStemming")
    pre_tool.stemming()
    print(pre_tool.corpus)

    print("\nLemming")
    pre_tool.lemmatization()
    print(pre_tool.corpus)