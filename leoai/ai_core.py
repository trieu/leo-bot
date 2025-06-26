import os
import logging
from dotenv import load_dotenv
# Imports from the Google AI SDK
from google import genai
from google.genai import types
from google.genai.types import GenerationConfig, HarmCategory, HarmBlockThreshold, Schema
from google.api_core.exceptions import GoogleAPIError
import json
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default fallback values
DEFAULT_MODEL_ID = os.getenv("GEMINI_TEXT_MODEL_ID", "gemini-2.0-flash-001")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Error message used when required config is missing
INIT_FAIL_MSG = "Both `model_name` and `api_key` must be provided or set in the environment."
JSON_TYPE = "application/json"


def is_gemini_model_ready():
    isReady = isinstance(GEMINI_API_KEY, str)
    # init Google AI
    if isReady:
        return True
    else:
        return False

class GeminiClient:
    """
    A wrapper class for interacting with Google Gemini API.
    Handles text generation via a specified model.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_ID, api_key: str = GEMINI_API_KEY):
        if not model_name or not api_key:
            logger.critical(INIT_FAIL_MSG)
            raise ValueError(INIT_FAIL_MSG)

        self.model_name = model_name
        self.api_key = api_key

        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Gemini client initialized with model '{self.model_name}'")
        except Exception as e:
            logger.exception("Failed to initialize Gemini client.")
            raise

    # text to text
    def generate_content(self, prompt: str, temperature: float = 0.6, on_error: str = '') -> str:
        """
        Generates text content from a given prompt using the Gemini API.

        Args:
            prompt (str): The input prompt to send to the model.
            temperature (float): Sampling temperature for creativity.
            on_error (str): Fallback string if generation fails.

        Returns:
            str: Generated content or fallback string.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature
                ),
            )
            if response.candidates and response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text.strip()
                logger.info(f"Generated content: {text}")
                return text
            else:
                logger.warning("Empty response received from Gemini API.")
                return on_error

        except GoogleAPIError as e:
            logger.error(f"Google API error during content generation: {e}")
            return on_error
        except Exception as e:
            logger.exception("Unexpected error during content generation.")
            return on_error
        
    # text to JSON
    def generate_json(self, prompt: str, json_schema: Schema) -> Dict[str, Any]:
        """
        Generates a structured JSON object from a prompt based on a provided schema.
        
        Args:
            prompt: The input prompt for the model.
            json_schema: The google.generativeai.types.Schema defining the desired JSON output.

        Returns:
            A dictionary parsed from the model's JSON response, or an empty dict on error.
        """
        try:
            # Configure the model for JSON output mode with the specified schema
            
            generation_config = GenerationConfig(
                response_mime_type=JSON_TYPE,
                response_schema=json_schema
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
            )

            response_text = response.text.strip()
            return json.loads(response_text)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from model response: {e}")
            logger.debug(f"Raw model response: {response.text if 'response' in locals() else 'N/A'}")
            return {}
        except GoogleAPIError as e:
            logger.error(f"A Google API error occurred: {e}")
            return {}
        except Exception as e:
            logger.exception(f"An unexpected error occurred in generate_json: {e}")
            return {}
