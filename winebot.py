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

from collections import Counter
import numpy as np
import nltk
import re
import sklearn.manifold
import multiprocessing
import pandas as pd
import gensim.models.word2vec as w2v
import csv
import json

#read in the file
data = pd.read_json(r'/Users/anuheaparker/Desktop/portfolio/wine_rec/winemag-data-130k-v2.json', dtype={
    'points': np.int32,
    'price': np.float32,
})


labels = data['variety']
descriptions = data['description']
wine_title = data['title']

#will count up how many of each type of wine (ex. albarino) is in the dataset
#varietal_counts = labels.value_counts()
#print(varietal_counts[:50])

#function for pairing a wine with a given food
def food_pairing(food):
	print("this is the food you want to pair:", food)


#function for giving a wine recommendation based on a taste description
def taste_description(taste):
	print("this is the description you want to use", taste)


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
    
