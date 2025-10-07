import  datetime
import logging
from typing import Dict, Literal, Optional
import markdown
import os
from dotenv import load_dotenv
load_dotenv()


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


# Define a type for format choices for better type hinting
AnswerFormat = Literal['html', 'text']

# Easier to read and modify. The [INST] tags are removed as they are not standard for Gemini.
PROMPT_TEMPLATE = """Your name is LEO, a helpful AI assistant.
Your response must be in the language: {target_language}.

Use the following context to answer the question.
Context: {context}
Question: {question}
"""

def _build_prompt(question: str, context: str, target_language: str) -> str:
    """Builds the full prompt string from the template."""
    current_time_str = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    
    # More Pythonic check for context
    if context:
        full_context = f"Current date and time is {current_time_str}. {context}"
    else:
        full_context = f"Current date and time is {current_time_str}. The user provided no additional context."
        
    return PROMPT_TEMPLATE.format(
        target_language=target_language,
        context=full_context,
        question=question
    )

def _format_response(answer_text: str, format: AnswerFormat) -> str:
    """Formats the raw model output into the desired format."""
    if format == 'html':
        # Using extensions for features like code blocks and tables is great
        return markdown.markdown(answer_text, extensions=['fenced_code', 'tables'])
    # Default to returning the raw text
    return answer_text

# the main function to ask LEO
def ask_question(
    context: str = '',
    question: str = 'Hi',
    answer_format: AnswerFormat = 'text',
    target_language: str = 'Vietnamese',
    temperature_score: float = TEMPERATURE_SCORE,
    gemini_client: Optional[GeminiClient] = None,
) -> str:

    """ Asks a question to the Gemini model and returns a formatted, translated answer.

    Args:
        context (str, optional): Additional context for the question. Defaults to ''.
        question (str, optional): The user's question.. Defaults to 'Hi'.
        answer_format (AnswerFormat, optional): The desired output format ('html' or 'text'). Defaults to 'text'.
        target_language (str, optional): The language code for the response (e.g., 'English', 'Vietnamese'. Defaults to 'Vietnamese'.
        temperature_score (float, optional): The creativity of the model's response.. Defaults to TEMPERATURE_SCORE.
        gemini_client (Optional[GeminiClient], optional): the instance of GeminiClient. Defaults to None.

    Returns:
        str: A formatted and translated string, or an error message.
    """
    
    try:
        # Initialize the generative model client
        client = gemini_client or GeminiClient()
        final_prompt = _build_prompt(question, context, target_language)
    
        logger.info(f"Generated prompt for model:\n{final_prompt}")
        raw_answer = client.generate_content(final_prompt, temperature_score)
        
        if not raw_answer:
            logger.warning("Model returned an empty response.")
            raw_answer = "I'm sorry, I couldn't generate a response for that."

    # Handle other potential exceptions
    except Exception as e:
        # Log the full error for debugging, which is more useful than just printing
        logger.exception(f"An unexpected error occurred while calling the AI model: {e}")
        
        question_query = question.replace(" ", "+")
        # Fallback message is now a raw string, letting _format_response handle it
        raw_answer = (
            "That's an interesting question. I encountered an issue and can't answer right now, "
            f"but you can try searching for it on [Google](https://www.google.com/search?q={question_query})."
        )

    return _format_response(raw_answer, answer_format)


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
    
