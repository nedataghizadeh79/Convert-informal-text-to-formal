from hazm import POSTagger, Normalizer
from enum import Enum, auto
from hazm import SentenceTokenizer, WordTokenizer
import itertools
import pickle


"""
each sentence will be associated with a vector of len(Tag) elements, each element representing the number of words with that tag in the sentence
(N, Ne, V, Aj, Adv, Det, P, Conj, Pron, Num, Postp, Prep, Interj, Abbrev, Aux, Punc)
example: بابا آب داد
(2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0, 0, 0, 0, 0)
"""

class Tag(Enum):
    NOUN = auto()
    NE = auto()
    VERB = auto()
    ADJ = auto()
    ADV = auto()
    ADP = auto()
    DET = auto()
    P = auto()
    CONJ = auto()
    PRON = auto()
    NUM = auto()
    POSTP = auto()
    PREP = auto()
    INTJ = auto()
    ABBREV = auto()
    AUX = auto()
    PUNCT = auto()
    SCONJ = auto()
    CCONJ = auto()



class PosShufflerPattern():

    def __init__(self, pos_tagger, load_from_file=True):
        self.tagger = pos_tagger
        if load_from_file:
            self.persist_data("load")
        else:
            self.persist_data("build")


    def get_possible_permutations(self, sentence: str, limit=10):
        words = WordTokenizer().tokenize(sentence)
        tagged_words = self.tagger.tag(words)
        permutations = []
        sentence_vector = [0] * len(Tag)
        for word, tag in tagged_words:
            pure_tag = tag.split(',')[0] # some tags are followed by ,EZ so we ignore it!
            tag_index = Tag[pure_tag].value - 1
            sentence_vector[tag_index] += 1

        vector_additions = [[0]*len(sentence_vector)] + [[0]*i + [1] + [0]*(len(sentence_vector)-i-1) for i in range(len(sentence_vector))]
        for addition, vector_addition in enumerate(vector_additions):
            vector = [x + y for x, y in zip(sentence_vector, vector_addition)]
            if tuple(vector) in self.tagged_dataset:
                proper_sentence_orders = self.tagged_dataset[tuple(vector)]
                sorted_sentence_orders = sorted(proper_sentence_orders.items(), key=lambda item: item[1], reverse=True)
                
                # build all permutations of the sentence
                all_permutations = list(itertools.permutations(tagged_words))
                for proper_order, _ in sorted_sentence_orders:
                    if addition > 0:
                        the_order = tuple(filter(lambda x: x != Tag(addition).name , proper_order))
                    else:
                        the_order = proper_order
                    permutations += filter(lambda x: tuple([i[1].split(',')[0] for i in x]) == the_order, all_permutations)
                    if len(permutations) >= limit:
                        break

        return [tuple(item[0] for item in sublist) for sublist in permutations][:limit]


    def tag_dataset(self, resource='resources/Datasets/MirasText_sample.txt'):
        with open(resource, 'r') as f:
            text = f.read()
        sentences = SentenceTokenizer().tokenize(text)
        dataset = {}
        for sentece in tqdm(sentences):
            words = WordTokenizer().tokenize(sentece)
            tagged_words = self.tagger.tag(words)

            sentence_vector = [0] * len(Tag)
            for word, tag in tagged_words:
                pure_tag = tag.split(',')[0] # some tags are followed by ,EZ so we ignore it!
                tag_index = Tag[pure_tag].value - 1
                sentence_vector[tag_index] += 1

            sentence_parts_order = tuple([tag.split(',')[0] for _, tag in tagged_words])

            if tuple(sentence_vector) in dataset:
                if sentence_parts_order in dataset[tuple(sentence_vector)]:
                    dataset[tuple(sentence_vector)][sentence_parts_order] += 1
                else:
                    dataset[tuple(sentence_vector)][sentence_parts_order] = 1
            else:
                dataset[tuple(sentence_vector)] = {sentence_parts_order: 1}

        return dataset


    def persist_data(self, dataset_action="load"):
        if dataset_action == "build":
            self.tagged_dataset = self.tag_dataset()
            dataset_action = "save"

        if dataset_action == "save" and self.tagged_dataset:
            with open("resources/Datasets/trained_postag_orders.pickle", "wb") as f:
                pickle.dump(tagged_dataset, f)
        elif dataset_action == "load":
            with open("resources/Datasets/trained_postag_orders.pickle", "rb") as f:
                self.tagged_dataset = pickle.load(f)
