import snowballstemmer 

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

#TEST SNOWBALLSTEMMER
stemmer = snowballstemmer.stemmer('english')
print(stemmer.stemWords("connecting connection connected".split()))

#TEST NLTK + STOPWORDS
text = "hi my name is ashley. i like to party. my brother likes dogs."
stopWords = set(stopwords.words('english'))
words = word_tokenize(text)
wordsFiltered = [w for w in words if w not in stopWords]

print(wordsFiltered)