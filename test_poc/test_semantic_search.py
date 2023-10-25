from sentence_transformers import SentenceTransformer, util

from chromadb.utils import embedding_functions
import chromadb
import hashlib

import os

MODEL_NAME = 'sentence-transformers/msmarco-distilroberta-base-v2'
model = SentenceTransformer(MODEL_NAME)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)


# Corpus with example sentences
corpus_list = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.',
          'God is love.'
          ]

corpus_ids = []
corpus_embeddings = []

enums = enumerate(corpus_list)
for id, corpus in enums:
    corpus_embedding = model.encode(corpus, convert_to_tensor=True)
    corpus_embeddings.append(corpus_embedding)    
    corpus_ids.append(id)
    print( str(id) + " = " + corpus)    

# How to override an old sqlite3 module with pysqlite3 https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
chroma_client = chromadb.Client()
#collection = chroma_client.get_or_create_collection(name="test",embedding_function=sentence_transformer_ef)
#collection.add(documents=corpus_list, ids=corpus_ids, embeddings=corpus_embeddings)

# Query sentences:
queries = ['God is the path','A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']

# Find the closest 5 sentences of the corpus_list for each query sentence based on cosine similarity
top_k = min(5, len(corpus_list))
for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus_list:")

    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        corpus_id = hit['corpus_id']
        print("corpus_id: ", corpus_id)
        print(corpus_list[corpus_id], "(Score: {:.4f})".format(hit['score']))