import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams

from sentence_transformers import SentenceTransformer

import cityhash

collection_name = "cities_data"

def hash_string(string):
    """Hashes a string into an unsigned 64-bit integer (CityHash64 algorithm)."""
    return cityhash.CityHash64(string) 

def init_data():
    first_collection = client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_DIM_SIZE, distance=Distance.COSINE)
    )
    print(first_collection)

    
    file_path = './data/top_cities_vietnam.json'
    cities_data = read_json_file(file_path)
    if cities_data:
        # Now you have the data as a list
        for city in cities_data:
            id = hash_string(json.dumps(city))
            corpus = ' '.join(city['travelTypes']) + " - " + city['name'] + " - " + city['description'] 
            city_embedding = model.encode(corpus, convert_to_tensor=True).tolist()
            print(
                f"City: {city['name']}, Latitude: {city['latitude']}, Longitude: {city['longitude']}, id {id}")
            # Add more fields as needed
            operation_info = client.upsert(
                collection_name=collection_name,
                wait=True,
                points=[PointStruct(id=id, vector=city_embedding,
                                    payload={"city": city})]
            )

# Initialize the client
client = QdrantClient("localhost", port=6333)  # For production
client.set_model("BAAI/bge-base-en")

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

model = SentenceTransformer(MODEL_NAME)
VECTOR_DIM_SIZE = model.get_sentence_embedding_dimension()


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None





init_data()


query = 'Island Adventures'
query_embedding = model.encode(query, convert_to_tensor=True).tolist()
search_result = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=5
)

print('\n\n\n')
for rs in search_result:
    print(rs.payload['city']['name'])
