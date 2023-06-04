from transformer_based_model import TransformerBasedModel

myTransformer = TransformerBasedModel(k=100)

score = myTransformer.calculate_sentence_score('ماشین ندا آبی است')
print(score)

score = myTransformer.calculate_sentence_score('ماشین من قرمز است')
print(score)

print(myTransformer.find_ordered_sentence_using_some_permutations([
    'دمپایی او پاره بود',
    'بود پاره دمپایی او',
    'پاره بود دمپایی او',
    'او دمپایی پاره بود',
    'بود دمپایی او پاره',
]))

print(myTransformer.find_ordered_sentence_using_all_permutations('من تو را دوست دارم'))