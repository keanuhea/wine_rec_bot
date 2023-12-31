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
***************************************************************************************/"""

from collections import Counter, OrderedDict
import numpy as np
import nltk
import re
import multiprocessing
import pandas as pd
import gensim.models.word2vec as w2v
import csv
import json

from gensim.models import Word2Vec


#read in the file
data = pd.read_json(r'/Users/anuheaparker/Desktop/wine_bot/wine_rec/wine_rec_bot/winemag-data-130k-v2.json', dtype={
    'points': np.int32,
    'price': np.float32,
})


labels = data['variety']
descriptions = data['description']
wine_title = data['title']

description_test = descriptions[0:5]
wine_title_test = wine_title[0:5]

print(description_test)
print(wine_title_test)


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

