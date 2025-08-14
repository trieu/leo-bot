from llama_cpp import Llama
import sqlite3

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

# === STEP 2: Setup test SQLite DB ===
conn = sqlite3.connect(":memory:")  # in-memory DB for testing
cur = conn.cursor()

# Example table
cur.execute("""
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary INTEGER
)
""")
cur.executemany("""
INSERT INTO employees (name, department, salary)
VALUES (?, ?, ?)
""", [
    ("Alice", "Engineering", 120000),
    ("Bob", "Sales", 90000),
    ("Charlie", "Engineering", 110000),
    ("Diana", "HR", 80000)
])
conn.commit()

# === STEP 3: SQL Agent prompt ===
def generate_sql(user_request: str) -> str:
    system_prompt = (
        "You are a helpful assistant that converts natural language questions into SQL queries. "
        "Only output SQL without explanations. The database is SQLite and follows standard SQL."
    )
    prompt = f"[INST] {system_prompt}\nUser request: {user_request} [/INST]"

    output = llm(
        prompt,
        max_tokens=256,
        temperature=0,
        stop=["[/INST]", "</s>"]
    )
    return output["choices"][0]["text"].strip()

# === STEP 4: Agent loop ===
while True:
    question = input("\nAsk a question about employees (or 'exit'): ")
    if question.lower() == "exit":
        break

    sql_query = generate_sql(question)
    print("\nGenerated SQL:\n", sql_query)

    try:
        cur.execute(sql_query)
        results = cur.fetchall()
        print("Results:", results)
    except Exception as e:
        print("SQL execution error:", e)

conn.close()