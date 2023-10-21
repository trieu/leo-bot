
from transformers import pipeline


MODEL_NLP = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipe = pipeline("sentiment-analysis", model=MODEL_NLP, tokenizer=MODEL_NLP)

