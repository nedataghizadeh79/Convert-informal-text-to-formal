import text_formalizer


informal_text = """
.علی و من رفتیم به کتابخانه
من از کتابخانه کتابی را گرفتم.
علی دنبالم به آمد.
رفتم دانشگاه.
از دانشگاه رفتم خانه.
"""

print(text_formalizer.formalize(informal_text))