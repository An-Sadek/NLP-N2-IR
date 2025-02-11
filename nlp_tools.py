import nltk
nltk.download('stopwords')

import regex as re

from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
import string


class Preprocessing:

    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self.pattern = r"(?:#|@)\w+\s*"

    def remove_hashtag_username(self):
        self.corpus = [re.sub(self.pattern, '', text).strip() for text in self.corpus]

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
        
    
        

    
if __name__ == "__main__":
    corpus = [
        "The mechanism of pattern recognition in the brain is \
        little known, and it seems to be almost impossible to \
        reveal it only by conventional physiological experiments. #A", 
        
        "So, we take a slightly different approach to this. #A @B"]
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