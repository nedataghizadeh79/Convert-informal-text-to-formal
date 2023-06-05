from hazm import POSTagger, Normalizer
from enum import Enum, auto
from hazm import SentenceTokenizer, WordTokenizer
import itertools
import pickle


class PosShufflerNGram():

    def __init__(self, pos_tagger, N=3, load_from_file=True):
        self.N = N
        self.tagger = pos_tagger
        if load_from_file:
            self.persist_data("load")
        else:
            self.persist_data("build")


    def get_possible_permutations(self, sentence: str, limit=10):
        words = WordTokenizer().tokenize(sentence)
        tagged_words = self.tagger.tag(words)
        permutations = itertools.permutations(tagged_words)
        all_possibilities = [(permutation, self.calculate_probability_of_postags([tag for word, tag in permutation])) for permutation in permutations]
        sorted_possibilities = sorted(all_possibilities, key=lambda x: x[1], reverse=True)[:limit]
        return [tuple(item[0] for item in sublist) for sublist in [x[0] for x in sorted_possibilities]]


    def calculate_probability_of_postags(self, postags: list):
        probability = 1
        tags = ['START'] + postags + ['END']
        for i in range(len(tags) - (self.N-1)):
            probability *= self.probability_dic.get(tuple(tags[i:i+self.N]), 1)
        return probability


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

    def persist_data(self, dataset_action="load"):
        if dataset_action == "build":
            self.probability_dic = get_probability_dic(n)
            dataset_action = "save"

        if dataset_action == "save" and probability_dic is not None:
            with open("resources/Datasets/ngram_shuffler.pickle", "wb") as f:
                pickle.dump(self.probability_dic, f)
        elif dataset_action == "load":
            with open("resources/Datasets/ngram_shuffler.pickle", "rb") as f:
                self.probability_dic = pickle.load(f)
