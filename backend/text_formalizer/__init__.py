import hazm
from text_formalizer.ngram_pos_shuffler import PosShufflerNGram
from text_formalizer.pattern_pos_shuffler import PosShufflerPattern
from text_formalizer.transformer_based_model import TransformerBasedModel
from text_formalizer.ngram_based_model import NgramBasedModel
from tqdm import tqdm

posTagger = hazm.POSTagger(model='resources/pos_tagger.model')


def formalize(informal_text: str):
    sentences = hazm.SentenceTokenizer().tokenize(informal_text)
    transformer_result = []
    ngram_result = []

    for sentence in tqdm(sentences):
        ngram_pos_shuffler = PosShufflerNGram(
            posTagger, N=3, load_from_file=True)
        ngram_permutations = ngram_pos_shuffler.get_possible_permutations(
            sentence, limit=10)

        pattern_pos_shuffler = PosShufflerPattern(
            posTagger, load_from_file=True)
        pattern_permutations = pattern_pos_shuffler.get_possible_permutations(
            sentence, limit=10)

        permutations = list(set(ngram_permutations + pattern_permutations))
        permutation_sentences = [' '.join(permutation)
                                 for permutation in permutations]

        transformer_model = TransformerBasedModel()
        tresult = transformer_model.find_ordered_sentence_using_some_permutations(permutation_sentences, print_data=True)
        transformer_result.append(tresult)

        ngram_model = NgramBasedModel()
        nresult = ngram_model.find_ordered_sentence_using_some_permutations(permutation_sentences, print_data=True)
        ngram_result.append(nresult)

    transformer_formalized_text = '\n'.join(transformer_result)
    ngram_formalized_text = '\n'.join(ngram_result)

    return transformer_formalized_text, ngram_formalized_text
