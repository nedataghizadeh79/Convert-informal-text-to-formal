# NLP_HW3
## Convert informal text to formal.

#### Saee Saadat 97110263
#### Neda Taghizadeh Serajeh 98170743
#### Maryam Sadaat Razavi Taheri 98101639

<hr>

## POSTAG shuffler:
creates permutations of the tokens in the sentence based on their pos-tags

notebooks:
- N-GRAM based -> POS_Shuffler_ngram.ipynb
- Pattern based -> POS_shuffler.ipynb

<br>

## Transformer based language model:
Using bert, we can give a score to each permutation of the sentence. the permutation with the highest score is more likely to be correct.
notebook: transformer_based_model.ipynb

<br>

## NGram based language model:
same as Transformer based, but uses n-gram (3-gram)
notebook: ngram_based_model.ipynb

<br> 

The entire pipeline has been implemented in the text_formalizer module and it's usage is demonstrated in main.py.

Also to cut down on training time, most models have saved their trained dictionaries or ... in a file which loads them when needed. This is done in the text_formalizer module as well as the notebooks.

Other notebooks can be ignored. They were used for testing and or learning purposes.
