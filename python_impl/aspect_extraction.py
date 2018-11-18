import nltk

from nltk import word_tokenize, wordnet
from nltk.corpus import sentiwordnet


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


def pos_tagging(review_tokens_dict):
    pos_tags = {}
    for index, token in review_tokens_dict.items():
        pos_tags[index] = nltk.pos_tag(nltk.word_tokenize(token))

    return pos_tags


def is_noun(pos_tag):
    return pos_tag == 'NN' or pos_tag == 'NNP'


def extract_aspects(pos_tag_dict):
    prev_word = ''
    prev_tag = ''
    curr_word = ''
    aspects = []
    aspect_dict = {}

    for index, pos_tags in pos_tag_dict.items():
        for word, pos_tag in pos_tags:
            if is_noun(pos_tag):
                if is_noun(prev_tag):
                    curr_word = prev_word + ' ' + word
                else:
                    aspects.append(prev_word.upper())
                    curr_word = word
            prev_word = curr_word
            prev_tag = pos_tag

    for aspect in aspects:
        if aspects.count(aspect) > 1 and aspect not in aspect_dict.keys():
            aspect_dict[aspect] = aspects.count(aspect)

    return sorted(aspect_dict.items(), key=lambda x: x[1], reverse=True)


def extract_opinions(reviews_dict, aspects):
    output_aspect_opinion_tuples = {}
    positive_words = {}
    negative_words = {"don't", "never", "nothing", "nowhere", "noone", "none", "not",
                      "hasn't", "hadn't", "can't", "couldn't", "shouldn't", "won't",
                      "wouldn't", "don't", "doesn't", "didn't", "isn't", "aren't", "ain't"}
    for aspect, no in aspects:
        aspect_tokens = word_tokenize(aspect)
        adj_count = 0
        for key, review in reviews_dict.items():
            is_negative_sentiment = False

            if sum([int(aspect_token in str(review).upper()) for aspect_token in aspect_tokens]) == len(aspect_tokens):
                for negative in negative_words:
                    is_negative_sentiment = True if not is_negative_sentiment and negative.upper() in str(
                        review).upper() else is_negative_sentiment

                output_aspect_opinion_tuples.setdefault(aspect, [0, 0, 0])

                for word, tag in review:
                    if is_adjective_or_adverb(tag):
                        adj_count += 1
                        if word not in positive_words:
                            is_positive_word = is_positive(word)
                            positive_words[word] = is_positive_word
                        else:
                            is_positive_word = positive_words[word]
                        if is_negative_sentiment and is_positive_word is not None:
                            is_positive_word = not is_positive_word
                        add_aspect_sentiment_weights(aspect, is_positive_word, output_aspect_opinion_tuples)
        if adj_count > 0:
            output_aspect_opinion_tuples[aspect][0] = ((output_aspect_opinion_tuples[aspect][0] / adj_count) * 100, 2)
            output_aspect_opinion_tuples[aspect][1] = ((output_aspect_opinion_tuples[aspect][1] / adj_count) * 100, 2)
            output_aspect_opinion_tuples[aspect][2] = ((output_aspect_opinion_tuples[aspect][2] / adj_count) * 100, 2)

    return output_aspect_opinion_tuples


def add_aspect_sentiment_weights(aspect, is_positive_word, output_aspect_opinion_tuples):
    if is_positive_word:
        output_aspect_opinion_tuples[aspect][0] += 1
    elif not is_positive_word:
        output_aspect_opinion_tuples[aspect][1] += 1
    elif is_positive_word is None:
        output_aspect_opinion_tuples[aspect][2] += 1


def is_adjective_or_adverb(tag):
    return tag == 'JJ' or tag == 'JJR' or tag == 'JJS' or tag == 'RB' or tag == 'RBR' or tag == 'RBS'


def is_positive(word):
    wordSynset = wordnet.synsets(word)
    if len(wordSynset) != 0:
        word = wordSynset[0].name()
        orientation = sentiwordnet.senti_synset(word)
        return orientation.pos_score() > orientation.neg_score()
