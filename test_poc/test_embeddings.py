import numpy as np

'''
Trong học máy, một embedding là một đại diện vector cho một đối tượng. 
Nó là một cách để mã hóa thông tin về đối tượng thành một vector có thể được sử dụng bởi các mô hình học máy.
Embedding thường được sử dụng để đại diện cho các từ hoặc cụm từ trong ngôn ngữ tự nhiên. 
Ví dụ, một embedding cho từ "cat" có thể là một vector có các thành phần đại diện cho các thuộc tính của con mèo, chẳng hạn như kích thước, màu sắc và loài.
'''
def embed(word:str ):
  return np.array([len(word), word.count("a"), word.count("b"), word.count("c")])
'''
Đoạn mã trên định nghĩa một hàm embed() ánh xạ một từ thành một vector có hai thành phần. 
Thành phần đầu tiên đại diện cho độ dài của từ, 
trong khi thành phần thứ hai đại diện cho số lần chữ "a", "b", "c" xuất hiện trong từ. 
Đây là một embedding vì nó đại diện cho các thuộc tính của từ.
'''
print(embed("cat")) # [3 1 0 1]

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/msmarco-distilroberta-base-v2')

#Our knowledges we like to encode
knowledges = ['I love cat', 'Cat hates dog']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(knowledges)

#Print the embeddings
for sentence, embedding in zip(knowledges, embeddings):
    print("Sentence:", sentence)
    print("Embedding.shape: " + str(embedding.shape[0]) )
    print("")

# Let's build a simple knowledge database and a semantic search engine 
question = 'What does kitten eat ?'
query_embedding = model.encode(question)

knowledges_database = knowledges + ['The 2 kittens are gray', 
                                    '1 kitten is yellow', 
                                    'Kitty is a small cat',
                                    'All kittens love to eat chicken pate']
knowledges_database_embedding = model.encode(knowledges_database)

print("Similarity:", util.dot_score(query_embedding, knowledges_database_embedding))

print("Question: ", question, " \n The answers: ")
hits = util.semantic_search(query_embedding, knowledges_database_embedding, top_k=2)
hits = hits[0]      #Get the hits for the first query
for hit in hits:
    id = hit['corpus_id']   
    print(knowledges_database[id], " (ID: {:g}) (Score: {:.4f})".format(id, hit['score']))

'''
Question:  What does kitten eat ?  
 The answers: 
All kittens love to eat chicken pate  (ID: 5) (Score: 0.7159)
Kitty is a small cat  (ID: 4) (Score: 0.5537)
'''