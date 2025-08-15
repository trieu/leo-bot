from llama_cpp import Llama
from arango import ArangoClient


# === STEP 1: Setup ArangoDB connection and data ===
# Initialize the ArangoDB client. Adjust these parameters if your setup is different.
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("test_db", username="root", password="12345678")

# Create a collection if it doesn't exist
if db.has_collection("employees"):
    employees_collection = db.collection("employees")
else:
    employees_collection = db.create_collection("employees")

# Clear existing data and insert new documents
employees_collection.truncate()
employees_collection.insert_many([
    {"name": "Alice", "department": "Engineering", "salary": 120000, "location": "New York", "full_time": True},
    {"name": "Bob", "department": "Sales", "salary": 90000, "location": "San Francisco", "full_time": True},
    {"name": "Charlie", "department": "Engineering", "salary": 110000, "location": "New York", "full_time": True},
    {"name": "Diana", "department": "HR", "salary": 80000, "location": "Boston", "full_time": True},
    {"name": "Eve", "department": "Marketing", "salary": 95000, "location": "Chicago", "full_time": True},
    {"name": "Frank", "department": "Engineering", "salary": 115000, "location": "Austin", "full_time": False},
    {"name": "Grace", "department": "Sales", "salary": 87000, "location": "San Francisco", "full_time": True},
    {"name": "Hank", "department": "Finance", "salary": 105000, "location": "New York", "full_time": True},
    {"name": "Ivy", "department": "Marketing", "salary": 98000, "location": "Boston", "full_time": False},
    {"name": "Jack", "department": "Engineering", "salary": 125000, "location": "Austin", "full_time": True}
])

# === STEP 2.1: download mistral-7b-instruct-v0.2.Q6_K.gguf
# at the https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main

class AQLAgent:
    def __init__(self, model_path: str, collection_name: str, fields: dict, n_ctx: int = 4096, n_threads: int = 16):
        """
        Initialize the AQL Agent with a local LLaMA model.
        
        :param model_path: Path to the GGUF model.
        :param collection_name: Name of the collection (e.g., "employees").
        :param fields: Dictionary of field_name: type (e.g., {"name": "string", "salary": "number"}).
        """
        self.collection_name = collection_name
        self.fields = fields
        print("Loading Mistral model...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            verbose=False
        )
    
    def _generate_system_prompt(self, user_request: str) -> str:
        """Create a structured system prompt for the LLM."""
        fields_list = ", ".join(f"{k} ({v})" for k, v in self.fields.items())
        examples = f"""
Examples:
User: show all employees in Engineering
AQL: FOR e IN {self.collection_name} FILTER e.department == "Engineering" RETURN e

User: get total salary
AQL: FOR e IN {self.collection_name} COLLECT AGGREGATE totalSalary = SUM(e.salary) RETURN totalSalary

User: get total salary for the Sales department
AQL: FOR e IN {self.collection_name} FILTER e.department == "Sales" COLLECT AGGREGATE totalSalary = SUM(e.salary) RETURN totalSalary

User: list employee names in Sales earning above 80000
AQL: FOR e IN {self.collection_name} FILTER e.department == "Sales" AND e.salary > 80000 RETURN e.name
"""
        prompt = f"""You are an expert in ArangoDB.
Given a user request, output ONLY the AQL query for the '{self.collection_name}' collection.
The fields are: {fields_list}.
{examples}
Now convert the request:
User: {user_request}
AQL:
"""
        return prompt

    def generate_aql(self, user_request: str) -> str:
        """Generate an AQL query using the LLM."""
        prompt = self._generate_system_prompt(user_request)
        output = self.llm(
            prompt,
            max_tokens=128,
            temperature=0,
            stop=["User:", "\n\n"]
        )
        return output["choices"][0]["text"].strip()
    
    def start_loop(self):
        # === STEP 4: Agent loop ===
        while True:
            question = input("\nAsk a question about employees (or 'exit'): ")
            if question.lower() == "exit":
                break

            aql_query = agent.generate_aql(question)
            print("\nGenerated AQL:\n", aql_query)

            try:
                cursor = db.aql.execute(aql_query)
                results = [doc for doc in cursor]
                print("Results:", results)
            except Exception as e:
                print("AQL execution error:", e)


# === Usage ===
model_file = "models/mistral-7b-instruct-v0.2.Q6_K.gguf"
fields = {"name": "string", "department": "string", "salary": "number"}
collection_name = "employees"
agent = AQLAgent(model_file, collection_name, fields)

query1 = agent.generate_aql("show all employees in Engineering")
query2 = agent.generate_aql("get total salary for Marketing department")

print(query1)
print(query2)

agent.start_loop()



