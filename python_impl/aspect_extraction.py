import nltk


def remove_stop_words(input_string):
    stop_words = nltk.corpus.stopwords.words("english")
    return ' '.join([word for word in input_string.split() if word not in stop_words])


def tokenize_reviews(review):
    tokens = {}
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    index = 1
    for sentence in tokenizer.tokenize(review):
        tokens[index] = sentence
        index += 1
    return tokens
