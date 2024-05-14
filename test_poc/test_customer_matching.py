import dedupe
import json, os
from unidecode import unidecode

# Define the fields dedupe will use
fields = [
    {'field' : 'Name', 'type': 'String'},
    {'field' : 'Address', 'type': 'String'},
    {'field' : 'Phone', 'type': 'String', 'has missing' : True},
]

# Create a new deduper object and pass our data model to it.
deduper = dedupe.Dedupe(fields)

# Sample JSON data with more than 5 records
data_json = '''
[
    {"ID": "1", "Name": "John Smith", "Address": "123 Main St", "Phone": "555-555-5555"},
    {"ID": "2", "Name": "Jon Smyth", "Address": "123 Main Street", "Phone": "555 555 5555"},
    {"ID": "3", "Name": "John Smith", "Address": "100 Main St", "Phone": "555.555.5555"},
    {"ID": "4", "Name": "Jane Smith", "Address": "123 Main St", "Phone": "555-555-5555"},
    {"ID": "5", "Name": "J. Smith", "Address": "123 Main St", "Phone": "555-555-5555"},
    {"ID": "6", "Name": "John Smith", "Address": "123 Main Street", "Phone": "555-555-5555"},
    {"ID": "7", "Name": "Johnny Smith", "Address": "100 Main St", "Phone": "555-555-5555"}
]
'''

# Convert JSON data to dictionary
data_d = {int(record['ID']): record for record in json.loads(data_json)}

# Preprocess the data
for record in data_d.values():
    record['Name'] = unidecode(record['Name'])
    record['Address'] = unidecode(record['Address'])
    record['Phone'] = unidecode(record['Phone']) if 'Phone' in record else None

# If we have training data saved from a previous run of dedupe,
# we load it here.
if os.path.exists('dedupe_training.json'):
    print('reading labeled examples from ', 'dedupe_training.json')
    with open('dedupe_training.json') as f:
        deduper.prepare_training(data_d, f)
else:
    deduper.prepare_training(data_d)

print('starting active labeling...')

dedupe.console_label(deduper)

deduper.train()

# Save our trained model to disk
with open('dedupe_settings', 'wb') as f:
    deduper.write_settings(f)

# Find duplicates
print('clustering...')
clustered_dupes = deduper.partition(data_d, 0.5)

print('reviewing duplicates...')
for cluster_id, (cluster, scores) in enumerate(clustered_dupes):
    print(cluster_id, cluster, scores)
