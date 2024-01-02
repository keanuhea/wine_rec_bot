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



#read in the file
data = pd.read_json(r'/Users/anuheaparker/Desktop/wine_bot/wine_rec/wine_rec_bot/winemag-data-130k-v2.json', dtype={
    'points': np.int32,
    'price': np.float32,
})

labels = data['variety']
descriptions = data['description']
wine_title = data['title']

description_test = descriptions[1:3]
wine_title_test = wine_title[1:3]

#print("this is description_test: ", description_test)

#varietal_counts = labels.value_counts()
#print(varietal_counts[0:50])

#FIRST STEP:
#Tokenize the data 

#concatenating all of description data into one big string
corpus_raw = ""
for description in description_test:
    corpus_raw += description
    
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(corpus_raw)


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

print("these are the finished sentences", sentences)




"""











#print(raw_sentences[1])
#print(sentence_to_wordlist(raw_sentences[1]))

#IMPORTANT: THIS WILL SPLIT WORDS WITH AN APOSTROPHE 
#SO ISN'T BECOMES "ISN" AND "T"

token_count = sum([len(sentence) for sentence in sentences])
print('The wine corpus contains {0:,} tokens'.format(token_count))

#can look into playing around with the min-word and context-size 
num_features = 300
min_word_count = 10
num_workers = multiprocessing.cpu_count()
context_size = 10
downsampling = 1e-3
seed=1993

wine2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    vector_size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

wine2vec.build_vocab(sentences)

print('Word2Vec vocabulary length:', len(wine2vec.wv))

print(wine2vec.corpus_count)

wine2vec.train(sentences, total_examples=wine2vec.corpus_count, epochs=wine2vec.epochs)


print(wine2vec.wv.most_similar('full'))

def nearest_similarity_cosmul(start1, end1, end2):
    similarities = wine2vec.wv.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

print(nearest_similarity_cosmul('oak', 'vanilla', 'cherry'))


"""











"""
tokenized_corpus = [review.lower().split() for review in description_test]

model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=2, sg=1, min_count=1)

# User input for taste description
user_description = input("Describe your taste preferences (e.g., dry, full-bodied, fruity): ").lower().split()

# Calculate the average embedding for the user's taste description
user_embedding = np.mean([model.wv[word] for word in user_description], axis=0)

# Calculate the average embedding for the user's taste description
valid_words = [word for word in user_description if word in model.wv]
if not valid_words:
    print("No valid words found in the vocabulary. Please provide different taste preferences.")
else:
    user_embedding = np.mean([model.wv[word] for word in valid_words], axis=0)
    


	# Calculate cosine similarity between user's taste and wine titles
similarities = [np.dot(user_embedding, model.wv[taste.lower()]) for taste in description_test]
    # Check if all similarities are NaN (no valid words found in the vocabulary)
if all(np.isnan(similarity) for similarity in similarities):
    print("No valid words found in the vocabulary. Please provide different taste preferences.")
else:
    recommended_wine_idx = np.nanargmax(similarities)


		# Output wine recommendation
recommended_wine = wine_title_test[recommended_wine_idx]
print(f"We recommend trying: {recommended_wine}")





#will count up how many of each type of wine (ex. albarino) is in the dataset
#varietal_counts = labels.value_counts()
#print(varietal_counts[:50])

#function for pairing a wine with a given food
def food_pairing(food):
	print("this is the food you want to pair:", food)


#function for giving a wine recommendation based on a taste description
def taste_description(taste):
	print("this is the description you want to use", taste)


"""


"""

#initial conversation with the user
def user_convo(search_style):

	if search_style.lower() == "food pairing":
		print("Great!")
		food = input("Please enter the name of the dish/ingredient you are going to eat: ")
		food_pairing(food)
	elif search_style.lower() == "taste description":
		print("Great!")
		taste = input("Please provide some taste descriptors (formatted with just a space in between each word): ")
		taste_description(taste)
	else:
		print("Sorry, I don't recognize your input. Please try again. ")
		search_style = input("How would you like to search for your perfect wine? Please type either 'food pairing' or 'taste description' to start your search process. ").strip()
		return user_convo(search_style)


if __name__ == "__main__":
    search_style = input("How would you like to search for your perfect wine? Please type either 'food pairing' or 'taste description' to start your search process. ").strip()
    user_convo(search_style)
    
"""

