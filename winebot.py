"""***************************************************************************************/

This code is adapted from Wine2Vec exploration by Zack Thoutt. I am also using Thoutt's dataset
from Kaggle (https://www.kaggle.com/datasets/zynicide/wine-reviews). While Thoutt is utilizing 
the wine reviews to recommend a wine from a description, I will be using this dataset in order to 
recommend a wine from a food pairing and vice versa. 


*    Title: Wine2Vec source code
*    Author: Thoutt, Z
*    Date: 2017
*    Availability: https://github.com/zackthoutt/wine-deep-learning/blob/master/Wine2Vec.ipynb
*

Here are some resources for the gensim library: 
https://github.com/piskvorky/gensim/wiki/Migrating-from-Gensim-3.x-to-4




***************************************************************************************/"""

from collections import Counter
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
#nltk.download('punkt')
#nltk.download('stopwords')
import re
import sklearn.manifold
import multiprocessing
import pandas as pd
import gensim.models.word2vec as w2v
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import snowballstemmer 
from sklearn.metrics.pairwise import cosine_similarity


#read in the file
data = pd.read_json(r'/Users/anuheaparker/Desktop/wine_bot/wine_rec/wine_rec_bot/winemag-data-130k-v2.json', dtype={
    'points': np.int32,
    'price': np.float32,
})

labels = data['variety']
descriptions = data['description']
wine_title = data['title']

description_test = descriptions[1:51]
wine_title_test = wine_title[1:51]

#varietal_counts = labels.value_counts()
#print(varietal_counts[0:50])


check = False
taste = ""
food = ""

if __name__ == "__main__":
    search_style = input("How would you like to search for your perfect wine? Please type either 'food pairing' or 'taste description' to start your search process. ").strip()
    
    while check == False:
    
        if search_style.lower() == "food pairing":
            print("Great!")
            food = input("Please enter the name of the dish/ingredient you are going to eat: ")
            check = True
        elif search_style.lower() == "taste description":
            print("Great!")
            taste = input("Please enter 1-3 taste descriptors: ")
            check = True
        else:
            print("Sorry, I don't recognize your input. Please try again. ")
            check = False
            search_style = input("How would you like to search for your perfect wine? Please type either 'food pairing' or 'taste description' to start your search process. ").strip()







#FIRST STEP:
#Tokenize the data 

#concatenating all of description data into one big string
corpus_raw = ""
for description in descriptions:
    #print("this is the description", description)
    corpus_raw += description
    
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#raw_sentences = tokenizer.tokenize(corpus_raw)
raw_sentences = descriptions


#SECOND STEP: 
#Clean the data. Remove punctuation, stopwords, and non-letters.

stop_words = set(stopwords.words('english'))
stemmer = snowballstemmer.stemmer('english')
punc_table = str.maketrans({key: None for key in string.punctuation})  



def sentence_to_wordlist(raw):
    try:
        word_list = word_tokenize(raw)
        norm_sentence = []
        for w in word_list:
            try:
                w = str(w)
                lower_case_word = w.lower()
                no_punc = re.sub("[^a-zA-Z]","", lower_case_word)
                stemmed_word = stemmer.stemWords(no_punc.split())
                clean_word = ' '.join(stemmed_word)
                if len(clean_word) > 1 and clean_word not in stop_words:
                    norm_sentence.append(clean_word)     
            except: 
                continue
        return norm_sentence
    except:
        return ''


sentences = []
for raw_sentence in raw_sentences:
    sentences.append(sentence_to_wordlist(raw_sentence))



# Tokenize user input (replace with your actual user input processing logic)
user_input = taste

user_input_tokenized = sentence_to_wordlist(user_input)

# Tokenize wine reviews and user input
all_reviews = sentences + [user_input_tokenized]

# Convert tokenized reviews to text strings
reviews_as_text = [" ".join(review) for review in all_reviews]

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
wine_vectors = vectorizer.fit_transform(reviews_as_text)

# Calculate cosine similarity
cosine_similarities = cosine_similarity(wine_vectors[:-1], wine_vectors[-1])

# Find the index of the most similar wine review
most_similar_index = cosine_similarities.argmax()



# Output wine recommendation
recommended_wine = wine_title[most_similar_index+1]
print(f"We recommend trying: {recommended_wine}")





