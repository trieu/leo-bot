from transformers import pipeline
from transformers import logging
logging.set_verbosity_error()

text_vi = "iPhone 15 chả có gì mới, Apple Watch vẫn chán như mọi khi"

# define model pipeline
model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipe = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)
lang_translation_pipe = pipeline("translation", model="facebook/m2m100_418M", 
              tokenizer="facebook/m2m100_418M", 
              src_lang="vi", tgt_lang="en")

# try test
text_en = lang_translation_pipe(text_vi)
print(text_en)
print(sentiment_pipe(text_en[0]['translation_text']))

text = '''As a product executive of more than 15 years, I find a lot of this lean product stuff to miss the mark. 
The book is okay for what it is, but lean product management doesn't really work well in practice except in particular UI based applications. 
And even then, companies that practice lean product management do a pretty terrible job 
with their interfaces given the amount of resources they have: examples in point: Amazon.com, Ebay, and Netflix.'''
print(sentiment_pipe(text))

#lang_detection_pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", device=0)
#print(lang_detection_pipe(text_vn))



