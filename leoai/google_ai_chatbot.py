import  datetime
import textwrap
import json
import os

from leoai.leo_datamodel import ChatMessage

from langchain.prompts import PromptTemplate

# need Google translate to convert input into English
from google.cloud import translate_v2 as translate
import pprint

# use Google AI
import markdown
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# to use local model "Mistral-7B", export LEOAI_LOCAL_MODEL=true
LEOAI_LOCAL_MODEL = os.getenv("LEOAI_LOCAL_MODEL") == "true"
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
TEMPERATURE_SCORE = 0.86

# default model names
GEMINI_1_5_MODEL = 'models/gemini-1.5-flash-latest'
GEMINI_1_5_PRO_MODEL = 'models/gemini-1.5-pro-latest'

# init PaLM client as backup AI
genai.configure(api_key=GOOGLE_GENAI_API_KEY)

# List all models
def list_models():
    for model in genai.list_models():
        pprint.pprint(model)

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

# the main function to ask LEO
def ask_question(context: str = '', answer_in_format: str = '', target_language: str = '', question: str = 'Hi', temperature_score = TEMPERATURE_SCORE ) -> str:
    context = context + '.Today, current date and time is ' + datetime.datetime.now().strftime("%c")
    template = """<s> [INST] Your name is LEO and you are the AI chatbot. 
    The response should answer for the question and context :
     {question} . {context} [/INST] </s>
    """
    prompt_tpl = PromptTemplate(template=template, input_variables=["question","context"])
    # set pipeline into LLMChain with prompt and llm model
    response = ""
    prompt_data = {"question":question,"context":context}
   
    prompt_text = prompt_tpl.format(**prompt_data)
    answer_text = 'No answer!'
    try:
        # call to Google Gemini APi
        gemini_text_model = genai.GenerativeModel(model_name=GEMINI_1_5_MODEL)
        model_config = genai.GenerationConfig(temperature=temperature_score)
        response = gemini_text_model.generate_content(prompt_text, generation_config=model_config)
        answer_text = response.text    
    except Exception as error:
        print("An exception occurred:", error)
        answer_text = "That's an interesting question."
        answer_text += "I have no answer by you can click here to check <a target='_blank' href='https://www.google.com/search?q=" + question + "'> "
        answer_text +=   "Google</a> ?"

    # translate into target_language 
    if isinstance(target_language, str) and isinstance(answer_text, str):
        if len(answer_text) > 1:
            rs = ''
            if answer_in_format == 'html':
                # format the answer in HTML
                answer_text = answer_text.replace('[LEO_BOT]', '[LEO_BOT]<br/>')
                # convert the answer in markdown into html 
                # See https://www.devdungeon.com/content/convert-markdown-html-python
                rs_html = markdown.markdown(answer_text, extensions=['fenced_code'])
                # translate into target language
                rs = translate_text(rs_html, target_language)
            else :
                answer_text = answer_text.replace('\n','<br/>')
                answer_text = answer_text.replace("```", "")                
                rs = translate_text(answer_text, target_language)
                rs = format_string_for_md_slides(rs)
            return rs
        else:
            return answer_text    
    elif answer_text is None:
        return translate_text("Sorry, I can not answer your question !", target_language) 
    # done
    return str(answer_text)


def extract_data_from_chat_message_by_ai(msg: ChatMessage) -> dict:
    content = msg.content
    if content is None:
        return {}
    prompt = textwrap.dedent("""\
            Return JSON describing the contacts, places, things from content using the following schema:

            {"contact": list[CONTACT], "places":list[PLACE], "order_details": list[ORDER_DETAILS]}

            CONTACT = {"first_name": str, "last_name": str, "description": str, "phone_number": str, "email": str, "address": str}
            PLACE = {"name": str, "description": str}
            ORDER_DETAILS = {"product_name": str, quality: int, value: int, description: str}

            All fields are required. Important: Only return a single piece of valid JSON text.
            Here is the content:

            """) + content
    try:
        gemini_model = genai.GenerativeModel(model_name=GEMINI_1_5_MODEL)
        response = gemini_model.generate_content(prompt, generation_config={'response_mime_type':'application/json'})

        # parse response into JSON
        extracted_data = json.loads(response.text)
        return extracted_data
    except Exception as error:
        print("An exception occurred:", error)
    
    return {}
    


