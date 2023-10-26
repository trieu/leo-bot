
import numpy as np

def embed(word:str ):
  return np.array([len(word), word.count("a"), word.count("b"), word.count("c")])

print(embed("cat"))

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/msmarco-distilroberta-base-v2')

#Our knowledges we like to encode
knowledges = ['I love cat', 'Cat hates dog', 'Kitty is a small cat']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(knowledges)

#Print the embeddings
for sentence, embedding in zip(knowledges, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

question = 'What does kitten eat ?'
query_embedding = model.encode(question)

knowledges_database = knowledges + ['The 2 kittens are gray', '1 kitten is yellow', 'All kittens love to eat chicken pate']
knowledges_database_embedding = model.encode(knowledges_database)

print("Similarity:", util.dot_score(query_embedding, knowledges_database_embedding))

hits = util.semantic_search(query_embedding, knowledges_database_embedding, top_k=2)
hits = hits[0]      #Get the hits for the first query
for hit in hits:
    id = hit['corpus_id']
    print("corpus_id: ", id)
    print(knowledges_database[id], "(Score: {:.4f})".format(hit['score']))