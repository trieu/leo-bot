import  datetime
import logging
from typing import Dict
import markdown
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate

# need Google translate to convert input into English
from google.cloud import translate_v2 as translate

from leoai.leo_datamodel import ChatMessage
from leoai.ai_core import GeminiClient
from google.genai.types import Schema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# to use local model "Mistral-7B", export LEOAI_LOCAL_MODEL=true
LEOAI_LOCAL_MODEL = os.getenv("LEOAI_LOCAL_MODEL") == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEMPERATURE_SCORE = 0.86

gemini_client = GeminiClient()

# the main function to ask LEO
def ask_question(
    context: str = '',
    question: str = 'Hi',
    answer_in_format: str = 'text', # 'html' or 'text'
    target_language: str = 'Vietnamese',
    temperature_score: float = TEMPERATURE_SCORE
) -> str:
    """
    Asks a question to the Gemini model and returns a formatted, translated answer.

    Args:
        context: Additional context for the question.
        question: The user's question.
        answer_in_format: The desired output format ('html' or 'text').
        target_language: The language code for the response (e.g., 'English', 'Vietnamese').
        temperature_score: The creativity of the model's response.

    Returns:
        A formatted and translated string, or an error message.
    """
    # Augment context with the current date and time
    current_time_str = datetime.datetime.now().strftime("%c")
    
    if len(context) > 10:        
        full_context = f"Current date and time is {current_time_str}. {context}"
    else:
        full_context = f"No context, just focus on the Question"
    
    if len(target_language) == 0:
        target_language = 'Vietnamese'

    # Simplified prompt using an f-string
    prompt_text = f"""<s> [INST] Your name is LEO and you are the AI chatbot to answer all questions from user.    
    The answer must be in the language: {target_language}  
    Answer the following question based on the provided context.  
    Context: {full_context}
    Question: {question}
    [/INST] </s>"""

    try:
        # Initialize the generative model
        logger.info(prompt_text)
        answer_text = gemini_client.generate_content(prompt_text, temperature_score)

    # --- Updated Error Handling ---
    # Handle other potential exceptions
    except Exception as e:
        print(f"An unexpected exception occurred: {e}")
        question_query = question.replace(" ", "+")
        answer_text = (
            "That's an interesting question. I don't have an answer right now, "
            f"but you can <a target='_blank' href='https://www.google.com/search?q={question_query}'>check Google</a>."
        )
        # Directly return HTML for this specific error case
        return answer_text

    # --- Formatting and Translation ---
    if not answer_text:
        return "Sorry, I could not answer your question."

    # Apply final formatting based on the requested format
    if answer_in_format == 'html':
        # Convert markdown response to HTML
        return markdown.markdown(answer_text, extensions=['fenced_code', 'tables'])
    else: # 'text' or default
        # Simple text formatting for slides or plain text display
        return str(answer_text)


# Translates text into the target language.
def translate_text(text: str, target: str) -> dict:
    if text == "" or text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    result = translate.Client().translate(text, target_language=target)
    return result['translatedText']

# detect language
def detect_language(text: str) -> str:
    if text == "" or text is None:
        return "en"
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    result = translate.Client().detect_language(text)
    print(result)
    if result['confidence'] > 0.9 :
        return result['language']
    else : 
        return "en"

def format_string_for_md_slides(rs):
    rs = rs.replace('<br/>','\n')
    rs = rs.replace('##','## ')
    return rs

def extract_data_from_chat_message_by_ai(msg: ChatMessage) -> Dict:
    """
    Extracts structured data from a chat message using the GeminiClient.
    
    This function analyzes text (expected to be Vietnamese) and uses the GeminiClient
    to extract contacts, places, and order details into a structured JSON format.
    """
    if not msg.content:
        logger.warning("Received a ChatMessage with no content.")
        return {}
    
    # Define the precise schema for the extraction task
    extraction_schema = Schema(
        type="OBJECT",
        properties={
            'contacts': Schema(type="ARRAY", items=Schema(type="OBJECT", properties={
                'first_name': Schema(type="STRING"), 'last_name': Schema(type="STRING"),
                'description': Schema(type="STRING"), 'phone_number': Schema(type="STRING"),
                'email': Schema(type="STRING"), 'address': Schema(type="STRING")
            })),
            'places': Schema(type="ARRAY", items=Schema(type="OBJECT", properties={
                'name': Schema(type="STRING"), 'description': Schema(type="STRING")
            })),
            'order_details': Schema(type="ARRAY", items=Schema(type="OBJECT", properties={
                'product_name': Schema(type="STRING"), 'quantity': Schema(type="INTEGER"),
                'value': Schema(type="NUMBER"), 'description': Schema(type="STRING")
            }))
        }
    )

    prompt_text = f"""
        Analyze the following Vietnamese text and extract information into a JSON object.
        Adhere strictly to the provided JSON schema. If information for a field is not
        present, use an empty string or list.

        **Content to Analyze:**
        ---
        {msg.content}
        ---
    """

    try:
        # Initialize the client and call the specialized JSON generation method
        gemini_client = GeminiClient()
        extracted_data = gemini_client.generate_json(prompt_text, json_schema=extraction_schema)
        return extracted_data
    except Exception as e:
        logger.exception(f"Failed to extract data: {e}")
        return {}