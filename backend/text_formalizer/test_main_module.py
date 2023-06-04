from transformer_based_model import TransformerBasedModel

myTransformer = TransformerBasedModel(k=100)

score = myTransformer.calculate_sentence_score('ماشین ندا آبی است')
print(score)

score = myTransformer.calculate_sentence_score('ماشین من قرمز است')
print(score)

print(myTransformer.find_ordered_sentence_using_all_permutations('من تو را دوست دارم'))