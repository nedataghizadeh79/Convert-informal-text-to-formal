import text_formalizer

def run():
    informal_text = """
    علی و من رفتیم به کتابخانه.
    من از کتابخانه کتابی را گرفتم.
    علی دنبالم به آمد.
    رفتم دانشگاه.
    از دانشگاه رفتم خانه.
    """

    transformer_formal_text, ngram_formal_text = text_formalizer.formalize(informal_text)
    print("\n\n=======================================\n\n")
    print("Transformer based formal text:")
    print(transformer_formal_text)

    print("\nNgram based formal text:")
    print(ngram_formal_text)

    with open('output.txt', 'w') as f:
        f.write("Input text:\n")
        f.write(informal_text + "\n")
        f.write("\nTransformer based formal text:\n")
        f.write(transformer_formal_text + "\n")
        f.write("\nNgram based formal text:\n")
        f.write(ngram_formal_text + "\n")


if __name__ == '__main__':
    run()