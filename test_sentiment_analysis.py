from transformers import pipeline

model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_pipe = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)

text = "As a product executive of more than 15 years, I find a lot of this lean product stuff to miss the mark. The book is okay for what it is, but lean product management doesn't really work well in practice except in particular UI based applications. And even then, companies that practice lean product management do a pretty terrible job with their interfaces given the amount of resources they have: examples in point: Amazon.com, Ebay, and Netflix."
print(sentiment_pipe(text))
