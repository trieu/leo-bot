from llama_cpp import Llama
from arango import ArangoClient
import psutil

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
    {"name": "Alice", "department": "Engineering", "salary": 120000,
        "location": "New York", "full_time": True},
    {"name": "Bob", "department": "Sales", "salary": 90000,
        "location": "San Francisco", "full_time": True},
    {"name": "Charlie", "department": "Engineering",
        "salary": 110000, "location": "New York", "full_time": True},
    {"name": "Diana", "department": "HR", "salary": 80000,
        "location": "Boston", "full_time": True},
    {"name": "Eve", "department": "Marketing", "salary": 95000,
        "location": "Chicago", "full_time": True},
    {"name": "Frank", "department": "Engineering",
        "salary": 115000, "location": "Austin", "full_time": False},
    {"name": "Grace", "department": "Sales", "salary": 87000,
        "location": "San Francisco", "full_time": True},
    {"name": "Hank", "department": "Finance", "salary": 105000,
        "location": "New York", "full_time": True},
    {"name": "Ivy", "department": "Marketing", "salary": 98000,
        "location": "Boston", "full_time": False},
    {"name": "Jack", "department": "Engineering",
        "salary": 125000, "location": "Austin", "full_time": True}
])

# === STEP 2.1: download mistral-7b-instruct-v0.2.Q6_K.gguf
# at the https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main


def compute_safe_n_ctx(
    model_size_gb: float,
    kv_per_token_mb: float = 0.5,
    runtime_overhead_gb: float = 1.0,
    safety_margin_gb: float = 0.5
) -> int:
    """
    Compute a safe n_ctx based on available RAM.

    :param model_size_gb: Size of model weights in GB (e.g., Q6_K Mistral-7B ≈ 6.4 GB)
    :param kv_per_token_mb: Approximate KV cache per token in MB for the model (default 0.5 MB/token for 7B)
    :param runtime_overhead_gb: Estimated runtime overhead for Python + llama.cpp in GB.
    :param safety_margin_gb: Extra safety margin to avoid swapping.
    :return: Safe n_ctx value (integer)
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)

    free_for_kv_gb = total_gb - \
        (model_size_gb + runtime_overhead_gb + safety_margin_gb)
    if free_for_kv_gb <= 0:
        raise MemoryError(
            "Not enough RAM for this model with given parameters.")

    free_for_kv_mb = free_for_kv_gb * 1024
    n_ctx = int(free_for_kv_mb / kv_per_token_mb)

    return max(256, n_ctx)  # enforce a reasonable minimum


MODEL_SIZE_GB = 6.4  # Mistral-7B Q6_K
DEFAULT_CONTEXT_SIZE = compute_safe_n_ctx(MODEL_SIZE_GB)
TOTAL_CPU_COUNT = psutil.cpu_count(logical=False)
DEFAULT_MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q6_K.gguf"


class AQLAgent:
    def __init__(self, collection_name: str, fields: dict, model_path: str = DEFAULT_MODEL_PATH, n_ctx: int = DEFAULT_CONTEXT_SIZE, n_threads: int = TOTAL_CPU_COUNT):
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
fields = {"name": "string", "department": "string", "salary": "number"}
collection_name = "employees"
agent = AQLAgent(collection_name, fields)

query1 = agent.generate_aql("show all employees in Engineering")
query2 = agent.generate_aql("get total salary for Marketing department")

print(query1)
print(query2)

agent.start_loop()
