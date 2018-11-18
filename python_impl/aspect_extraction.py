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
        if aspects.count(aspect) > 1 and  aspect not in aspect_dict.keys():
                aspect_dict[aspect] = aspects.count(aspect)

    return sorted(aspect_dict.items(), key=lambda x: x[1], reverse=True)
