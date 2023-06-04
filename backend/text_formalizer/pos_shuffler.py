import nltk
from nltk.util import ngrams
from hazm import POSTagger, Normalizer
from enum import Enum, auto
from hazm import SentenceTokenizer, WordTokenizer
import itertools
import pickle

N = 3
probability_dict = None


def make_n_gram_model(n, resource='resources/Datasets/MirasText_sample.txt'):
    with open(resource, 'r') as f:
        text = f.read()
    sentences = SentenceTokenizer().tokenize(text)
    n_grams = []
    for sentece in tqdm(sentences):
        words = WordTokenizer().tokenize(sentece)
        tagged_words = tagger.tag(words)

        # make 3-gram from tags
        tags = ['START'] + [tag for word, tag in tagged_words] + ['END']
        ng = ngrams(tags, n)
        n_grams.extend(ng)
    return n_grams


def get_probability_dic(n):
    probability_dic = {}
    ngram_model = make_n_gram_model(n)
    for tup in ngram_model:
        probability_dic[tup] = probability_dic.get(tup, 0.01) + 1
        if tup[0] == 'START':
            probability_dic[tup] += 100
    return probability_dic


def persist_data(dataset_action="load"):
    global probability_dict
    
    if dataset_action == "build":
        probability_dic = get_probability_dic(n)
        dataset_action = "save"

    if dataset_action == "save" and probability_dic is not None:
        with open("resources/Datasets/ngram_shuffler.pickle", "wb") as f:
            pickle.dump(probability_dic, f)
            print("Saved dataset to pickle file!")
    elif dataset_action == "load":
        with open("resources/Datasets/ngram_shuffler.pickle", "rb") as f:
            probability_dic = pickle.load(f)
            print("Loaded dataset from pickle file!")
