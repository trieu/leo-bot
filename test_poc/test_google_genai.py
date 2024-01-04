import google.generativeai as palm
import os
import markdown
from dotenv import load_dotenv
load_dotenv()

GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
palm.configure(api_key=GOOGLE_GENAI_API_KEY)

book_title = 'Big Data and Customer Data Platform In Real Estate'

question_for_toc = "Write a table of contents for the book '" + book_title + "'."
question_for_toc = question_for_toc + " Each chapter must begine with '##'"

section = "Automated data collection"
question_for_content = "You are the author of book '"+book_title+"'. Write a paragraph about '"+section+"'"

print("\n"+question_for_content+"\n")
result = palm.generate_text(prompt=question_for_content, temperature=0.8).result
rs_html = markdown.markdown(result, extensions=['fenced_code'])
print(result)

print("\n HTML: \n "+rs_html+"\n")

