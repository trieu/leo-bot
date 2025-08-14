from llama_cpp import Llama
from google.cloud import translate_v2 as translate

# need export GOOGLE_APPLICATION_CREDENTIALS=
def translate_text( text: str, target: str) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    translate_client = translate.Client()
    
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    return result['translatedText']

# Load GGUF model into llama.cpp
# download Phi-3-mini-4k-instruct-q4.gguf at https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/main
print("Loading model...")
llm = Llama(
    model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=4096,        # context length
    n_threads=16,      # use your 16 CPU cores
    n_batch=512,       # batch size for speed
    verbose=False
)

# Question
question = "How to get rich and make a lot of money"

# Phi-3 expects chat-style formatting
prompt = (
    f"<|system|>\nYou are a helpful assistant.\n" f"<|user|>\n{question}\n" f"<|assistant|>"
)

print("\nGenerating text...")
output = llm(
    prompt,
    max_tokens=800,
    temperature=0.8,
    top_p=0.9,
    stop=["<|user|>", "<|system|>"]
)

print("\n--- Generated Text ---")
final_text = translate_text(output["choices"][0]["text"].strip(),'vi')
print(final_text)