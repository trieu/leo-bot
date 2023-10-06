import google.generativeai as palm
import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
palm.configure(api_key=GOOGLE_GENAI_API_KEY)

question = "Write a story to use LEO Customer Data Platform with AI"

response = palm.generate_text(prompt=question)
print(response.result) 