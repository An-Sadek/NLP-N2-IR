# Load spacy and language
import spacy
nlp = spacy.load("en_core_web_sm")
words = set(nlp.vocab.strings)

# get string to check
string = "I am lkjdfosdaj fp9238u4213u4 bankai this is test text"

# check if each word is in the dictionary
problem_words = []
for word in string.split():
    if word not in words:
        problem_words.append(word)

# print the results
if problem_words:
    print("The following words are not in the dictionary:")
    for word in problem_words:
        print(word)