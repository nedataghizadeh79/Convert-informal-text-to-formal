import nltk
from nltk.util import ngrams
from collections import Counter
from collections import defaultdict
import pickle


class NgramBasedModel:
    def __init__(self):
        nltk.download('punkt')
        self.persist_dataset("load")


    def make_ngram_model(n, text):
        # Create a list of tokens from the text
        # Assuming 'text' contains the list of lines from the text file
        tokens = nltk.word_tokenize(' '.join(text))
        # Generate n-grams
        return list(ngrams(tokens, n))


    def make_dictionary_for_ngram_model(self, ngram_list):
        # Create a dictionary with default values of 0
        tuple_dict = defaultdict(int)

        # Count the occurrences of each tuple
        for tpl in ngram_list:
            tuple_dict[tpl] += 1

        # Divide all values by all number of the unique touple
        number_of_unique_touple = len(tuple_dict)
        for tpl in tuple_dict:
            tuple_dict[tpl] /= number_of_unique_touple

        return tuple_dict

    @staticmethod
    def divide_sentence(sentence, num):
        words = sentence.split()
        result = []
        
        for i in range(len(words) - num + 1):
            group = tuple(words[i:i+num])
            result.append(group)
        
        return result


    @staticmethod
    def calc_score_n_gram(word_n_groups , tuple_dict_for_n_gram):

        scores_dict = {}

        for group in word_n_groups:
            score = 1
            if group in tuple_dict_for_n_gram:
                score *= tuple_dict_for_n_gram[group]
            else:
                score = 0.000000000000000000000000000001
                scores_dict[group] = score

        final_score = 1
        for value in scores_dict.values():
            final_score *= value

        return final_score


    def find_ordered_sentence_using_some_permutations(self, sentences_list, print_data=False):
        scores = []
        for sentence in sentences_list:
            print("calculating score for sentence: ", sentence)
            score = self.calc_score_n_gram(self.divide_sentence(sentence, 3), self.tuple_dict_for_three_gram)
            scores.append((sentence, score))
        if print_data:
            scores.sort(key=lambda x: x[1], reverse=True)
            for score in scores:
                print(score)
        return max(scores, key=lambda x: x[1])[0]


    def persist_dataset(self, dataset_action="load"):
        if dataset_action == "build":
            file_path = 'resources/Datasets/MirasText_sample.txt'

            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().splitlines()

            self.three_gram_list = make_ngram_model(3, text)
            self.tuple_dict_for_three_gram = make_dictionary_for_ngram_model(
                three_gram_list)

            dataset_action = "save"

        if dataset_action == "save" and tuple_dict_for_three_gram:
            with open("resources/Datasets/ngram_dataset.pickle", "wb") as f:
                pickle.dump(self.tuple_dict_for_three_gram, f)
                print("Saved dataset to pickle file!")
        elif dataset_action == "load":
            with open("resources/Datasets/ngram_dataset.pickle", "rb") as f:
                self.tuple_dict_for_three_gram = pickle.load(f)
                print("Loaded dataset from pickle file!")
