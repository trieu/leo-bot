from sentence_transformers import SentenceTransformer, util
import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")

model_id = 'sentence-transformers/msmarco-distilroberta-base-v2'
model = SentenceTransformer(model_id)

# Corpus with example sentences
corpus = ['A man is eating food.',
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
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
collection.add(
    embeddings=corpus_embeddings,
    documents=corpus,
   
)

# Query sentences:
queries = ['God is the path','A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))