import snowballstemmer 

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


"""


#TEST SNOWBALLSTEMMER
stemmer = snowballstemmer.stemmer('english')
print(stemmer.stemWords("connecting connection connected".split()))

#TEST NLTK + STOPWORDS
text = "hi my name is ashley. i like to party. my brother likes dogs."
stopWords = set(stopwords.words('english'))
words = word_tokenize(text)
wordsFiltered = [w for w in words if w not in stopWords]

print(wordsFiltered)

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample tokenized wine reviews and titles (replace with your actual data)
wine_reviews_tokenized = [
    ["cabernet", "sauvignon", "rich", "full-bodied", "notes", "blackcurrant"],
    ["chardonnay", "dry", "white", "hint", "oak", "crisp", "finish"],
    ["merlot", "smooth", "velvety", "touch", "sweetness"],
    ["sparkling", "prosecco", "lively", "bubbles", "refreshing", "citrus", "flavors"],
    ["pinot", "noir", "light-bodied", "bright", "red", "fruit", "aromas", "silky", "texture"],
]

wine_titles = ["Cabernet Sauvignon", "Chardonnay", "Merlot", "Prosecco", "Pinot Noir"]

# User input (replace with your user input processing logic)
user_input_words = ["dry", "fruity", "bold"]

# Combine user input words into a single string
user_input_text = " ".join(user_input_words)

# Tokenize user input (replace with your actual user input processing logic)
user_input_tokenized = ["dry", "fruity", "bold"]

# Tokenize wine reviews and user input
all_reviews = wine_reviews_tokenized + [user_input_tokenized]

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
recommended_wine = wine_titles[most_similar_index]
print(f"We recommend trying: {recommended_wine}")