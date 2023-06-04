import tensorflow as tf
import itertools
from transformers import TFAutoModelForMaskedLM, AutoTokenizer


class TransformerBasedModel:
    def __init__(self, k=1000, epsilon=0.000001):
        print('loading model...')
        model_name = "HooshvareLab/bert-base-parsbert-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=False,
                                                            output_attentions=False)
        self.k = k
        self.epsilon = epsilon
        print('model loading done.')

    def calculate_sentence_score(self, sentence, k=None, epsilon=None):
        print('calculating score for', sentence, '...')
        if k is None:
            k = self.k
        if epsilon is None:
            epsilon = self.epsilon
        sentence_score = 1000000000000
        sentence_tokens = self.tokenizer.tokenize(sentence)
        for i, token in enumerate(sentence_tokens):
            found = False
            masked_text = ' '.join(sentence_tokens[:i]) + ' [MASK] ' + ' '.join(sentence_tokens[i + 1:])
            input_ids = self.tokenizer.encode(masked_text, return_tensors="tf")
            logits = logits = self.model.__call__(input_ids)[0]
            mask_token_index = tf.where(input_ids == self.tokenizer.mask_token_id)[0, 1]
            mask_token_logits = logits[0, mask_token_index, :]
            top_k_values, top_k_indices = tf.math.top_k(mask_token_logits, k=k)
            predicted_token_indices = top_k_indices.numpy()
            probabilities = tf.nn.softmax(top_k_values).numpy()
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_token_indices)
            for j, predicted_token in enumerate(predicted_tokens):
                if predicted_token == token:
                    sentence_score *= probabilities[j]
                    found = True
                    break
            if not found:
                sentence_score *= epsilon
        print('calculating score done.')
        return sentence_score

    def find_ordered_sentence_using_some_permutations(self, sentences_list, k=None, epsilon=None, print_data=False):
        if k is None:
            k = self.k
        if epsilon is None:
            epsilon = self.epsilon
        scores = []
        for i, sentence in enumerate(sentences_list):
            score = self.calculate_sentence_score(sentence, k, epsilon)
            if print_data:
                print(sentence)
                print(score)
                print(i + 1, '/', len(sentences_list))
            scores.append((sentence, score))
        if print_data:
            scores.sort(key=lambda x: x[1], reverse=True)
            for score in scores:
                print(score)
        return max(scores, key=lambda x: x[1])[0]

    def find_ordered_sentence_using_all_permutations(self, sentence, k=None, epsilon=None, print_data=False):
        if k is None:
            k = self.k
        if epsilon is None:
            epsilon = self.epsilon
        tokens = self.tokenizer.tokenize(sentence)
        permutations = list(itertools.permutations(tokens))
        scores = []
        for i, permutation in enumerate(permutations):
            permutation_sentence = ' '.join(permutation)
            score = self.calculate_sentence_score(permutation_sentence, k, epsilon)
            if print_data:
                print(permutation_sentence)
                print(score)
                print(i + 1, '/', len(permutations))
            scores.append((permutation_sentence, score))
        if print_data:
            scores.sort(key=lambda x: x[1], reverse=True)
            for score in scores:
                print(score)
        return max(scores, key=lambda x: x[1])[0]
