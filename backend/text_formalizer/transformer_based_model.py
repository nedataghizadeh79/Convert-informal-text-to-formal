import tensorflow as tf
import itertools
from transformers import TFAutoModelForMaskedLM, AutoTokenizer

model_name = "HooshvareLab/bert-base-parsbert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=False, output_attentions=False)


def calculate_sentence_score(sentence_tokens, k=1000, epsilon=0.000001):
    sentence_score = 1000000000000
    sentence = ' '.join(sentence_tokens)
    for i, token in enumerate(sentence_tokens):
        found = False
        masked_text = ' '.join(sentence_tokens[:i]) + ' [MASK] ' + ' '.join(sentence_tokens[i + 1:])
        input_ids = tokenizer.encode(masked_text, return_tensors="tf")
        logits = logits = model.__call__(input_ids)[0]
        mask_token_index = tf.where(input_ids == tokenizer.mask_token_id)[0, 1]
        mask_token_logits = logits[0, mask_token_index, :]
        top_k_values, top_k_indices = tf.math.top_k(mask_token_logits, k=k)
        predicted_token_indices = top_k_indices.numpy()
        probabilities = tf.nn.softmax(top_k_values).numpy()
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_indices)
        for j, predicted_token in enumerate(predicted_tokens):
            if predicted_token == token:
                sentence_score *= probabilities[j]
                found = True
                break
        if not found:
            sentence_score *= epsilon
    return sentence_score


def clean_permutations(permutations):
    return permutations


def find_ordered_sentence(sentence):
    tokens = tokenizer.tokenize(sentence)
    permutations = list(itertools.permutations(tokens))
    cleaned_permutations = clean_permutations(permutations)
    scores = []
    for i, permutation in enumerate(cleaned_permutations):
        permutation_sentence = ' '.join(permutation)
        score = calculate_sentence_score(list(permutation))
        print(permutation_sentence)
        print(score)
        scores.append((permutation_sentence, score))
        print(i + 1, '/', len(permutations))
    scores.sort(key=lambda x: x[1], reverse=True)
    for score in scores:
        print(score)
    return scores[0][1]


find_ordered_sentence('ماشین من قرمز است')
