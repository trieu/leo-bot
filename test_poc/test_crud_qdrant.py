import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams

from sentence_transformers import SentenceTransformer

import cityhash

CITIES_DATA = "cities_data"


def hash_string(string):
    """Hashes a string into an unsigned 64-bit integer (CityHash64 algorithm)."""
    return cityhash.CityHash64(string)


def init_data(file_path: str):
    first_collection = client.recreate_collection(
        collection_name=CITIES_DATA,
        vectors_config=VectorParams(
            size=VECTOR_DIM_SIZE, distance=Distance.COSINE)
    )
    print(first_collection)

    cities_data = read_json_file(file_path)
    if cities_data:
        # Now you have the data as a list
        for city in cities_data:
            id = hash_string(json.dumps(city))
            corpus = ' '.join(city['travelTypes']) + " - " + \
                city['name'] + " - " + city['description']
            city_embedding = model.encode(
                corpus, convert_to_tensor=True).tolist()

            print(
                f"City: {city['name']}, Latitude: {city['lat']}, Longitude: {city['lon']}, id {id}")
            # Add more fields as needed

            p = PointStruct(id=id, vector=city_embedding,
                            payload={"city": city})
            operation_info = client.upsert(
                collection_name=CITIES_DATA,
                wait=True,
                points=[p]
            )

    # indexing
    client.create_payload_index(
        collection_name=CITIES_DATA,
        field_name="city.population",
        field_schema="integer",
    )
    client.create_payload_index(
        collection_name=CITIES_DATA,
        field_name="city.travelTypes",
        field_schema="keyword",
    )
    client.create_payload_index(
        collection_name=CITIES_DATA,
        field_name="city",
        field_schema="geo",
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


def test_query():
    query = 'Any place with cool climate and sunny beach'
    r_in_km = 1000.0 * 1000
    query_embedding = model.encode(query, convert_to_tensor=True).tolist()

    search_result = client.search(
        collection_name=CITIES_DATA,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="city.population",
                    range=models.Range(
                        gt=100000,
                        gte=None,
                        lt=5000000,
                        lte=None
                    ),
                ),
                models.FieldCondition(
                    key="city.travelTypes",
                    match=models.MatchAny(any=["History","Nightlife"]),
                ),
                models.FieldCondition(
                    key="city",
                    geo_radius=models.GeoRadius(
                        center=models.GeoPoint(
                            lat=10.7619578,
                            lon=106.6873586
                        ),
                        radius=r_in_km,
                    ),
                )
            ]
        ),
        query_vector=query_embedding,
        limit=5
    )

    print('\n\n\n Query: ' + query)
    for rs in search_result:
        print(rs.payload['city']['name'])


# main start
file_path = './data/top_cities_vietnam.json'

# 1. create collecion and init data
init_data(file_path)

# 2. test
test_query()
