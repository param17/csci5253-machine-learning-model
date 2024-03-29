import re
import sys
import json
import nltk

nltk.download('stopwords')

from pandas import DataFrame
from nltk.corpus import stopwords
from collections import defaultdict
from textblob import TextBlob


def clean_dataset(dataset):
    cleaned_reviews = []
    cleaned_reviews_dict = defaultdict(list)
    stop_words = set(stopwords.words('english'))
    for i, sentence in enumerate(dataset):
        sentence = re.sub(r"n\'t", " not", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'s", " is", sentence)
        sentence = re.sub(r"\'t", " not", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'m", " am", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r" v", " very", sentence)
        sentence = re.sub('[^a-zA-Z.]', ' ', sentence)
        sentence = sentence.lower()
        sentence = sentence.split('.')
        for k in range(len(sentence)):
            review = sentence[k].split()
            review = [word.decode("utf-8") for word in review if not word in stop_words]
            sentence[k] = ' '.join(review)
            sentence[k] = TextBlob(sentence[k]).correct()
            cleaned_reviews_dict[i].append(sentence[k])
        cleaned_reviews.append(cleaned_reviews_dict[i])

    return cleaned_reviews


with open(sys.argv[0]) as dataset_file:
    dataset = json.load(dataset_file)

product_reviews = clean_dataset([dataset[index]['review'] for index in range(len(dataset))])

product_reviews_dataset = DataFrame({'reviewText': product_reviews})

