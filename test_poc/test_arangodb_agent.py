from llama_cpp import Llama
from arango import ArangoClient


# === STEP 0: download mistral-7b-instruct-v0.2.Q6_K.gguf
# at the https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main

# === STEP 1: Load model ===
print("Loading Mistral model...")
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q6_K.gguf",  # Local GGUF model
    n_ctx=4096,
    n_threads=16,  # Adjust to your CPU
    n_batch=512,
    verbose=False
)

# === STEP 2: Setup ArangoDB connection and data ===
# Initialize the ArangoDB client. Adjust these parameters if your setup is different.
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("test_db", username="root", password="12345678")

# Create a collection if it doesn't exist
# Create a collection if it doesn't exist
if db.has_collection("employees"):
    employees_collection = db.collection("employees")
else:
    employees_collection = db.create_collection("employees")

# Clear existing data and insert new documents
employees_collection.truncate()
employees_collection.insert_many([
    {"name": "Alice", "department": "Engineering", "salary": 120000},
    {"name": "Bob", "department": "Sales", "salary": 90000},
    {"name": "Charlie", "department": "Engineering", "salary": 110000},
    {"name": "Diana", "department": "HR", "salary": 80000}
])

# === STEP 3: AQL Agent prompt ===
def generate_aql(user_request: str) -> str:
    system_prompt = f"""You are an expert in ArangoDB.
Given a user request, output ONLY the AQL query for the 'employees' collection.
The fields are: name (string), department (string), salary (number).

Examples:
User: show all employees in Engineering
AQL: FOR e IN employees FILTER e.department == "Engineering" RETURN e

User: get total salary
AQL: FOR e IN employees COLLECT AGGREGATE totalSalary = SUM(e.salary) RETURN totalSalary

User: get total salary for the Sales department
AQL: FOR e IN employees FILTER e.department == "Sales" COLLECT AGGREGATE totalSalary = SUM(e.salary) RETURN totalSalary

User: list employee names in Sales earning above 80000
AQL: FOR e IN employees FILTER e.department == "Sales" AND e.salary > 80000 RETURN e.name

Now convert the request:
User: {user_request}
AQL:
"""
    output = llm(
        system_prompt,
        max_tokens=128,
        temperature=0,
        stop=["User:", "\n\n"]
    )
    return output["choices"][0]["text"].strip()


# === STEP 4: Agent loop ===
while True:
    question = input("\nAsk a question about employees (or 'exit'): ")
    if question.lower() == "exit":
        break

    aql_query = generate_aql(question)
    print("\nGenerated AQL:\n", aql_query)

    try:
        cursor = db.aql.execute(aql_query)
        results = [doc for doc in cursor]
        print("Results:", results)
    except Exception as e:
        print("AQL execution error:", e)