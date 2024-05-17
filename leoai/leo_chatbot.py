from datetime import date
import textwrap
import json
import os

from leoai.leo_datamodel import ChatMessage

# Local AI LLM Model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# need Google translate to convert input into English
from google.cloud import translate_v2 as translate
import pprint

import datetime

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
GEMINI_1_0_MODEL = 'models/gemini-1.0-pro-latest'
GEMINI_1_5_MODEL = 'models/gemini-1.5-pro-latest'

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

# local AI model
def load_llm_pipeline():
    # Quantize 🤗 Transformers models
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # define Mistral AI Large Language Models
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    model_4bit = AutoModelForCausalLM.from_pretrained( model_id, device_map="auto",quantization_config=quantization_config, )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # init pipeline of text-generation
    pipe = pipeline(
            "text-generation",
            model=model_4bit,
            tokenizer=tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=512,
            do_sample=True,        
            temperature=TEMPERATURE_SCORE,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
    )
    return pipe

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
llm_model = None
# check to init local HuggingFacePipeline, Minimum System Requirements: GeForce RTX 3060
if LEOAI_LOCAL_MODEL:
    llm_model = HuggingFacePipeline(pipeline=load_llm_pipeline())    
    torch.cuda.empty_cache() # clear GPU cache 

# the main function to ask LEO
def ask_question(context: str, answer_in_format: str, target_language: str, question: str, temperature_score = TEMPERATURE_SCORE ) -> str:
    context = context + '.Today, current date and time is ' + datetime.datetime.now().strftime("%c")
    template = """<s> [INST] Your name is LEO and you are the AI chatbot. 
    The response should answer for the question and context :
     {question} . {context} [/INST] </s>
    """
    prompt_tpl = PromptTemplate(template=template, input_variables=["question","context"])
    # set pipeline into LLMChain with prompt and llm model
    response = ""
    prompt_data = {"question":question,"context":context}
    if llm_model is not None:  
        llm_chain = LLMChain(prompt=prompt_tpl, llm=llm_model)
        response = llm_chain.run(prompt_data)

    src_text = f"{response}".strip()
    if len(src_text) == 0:
        prompt_text = prompt_tpl.format(**prompt_data)
        try:
            # call to Google AI PaLM 2 API
            gemini_text_model = genai.GenerativeModel(model_name=GEMINI_1_5_MODEL)
            model_config = genai.GenerationConfig(temperature=temperature_score)
            response = gemini_text_model.generate_content(prompt_text, generation_config=model_config)
            src_text = response.text    
        except Exception as error:
            print("An exception occurred:", error)
            src_text = "That's an interesting question.";
            src_text += "I have no answer by you can click here to check <a target='_blank' href='https://www.google.com/search?q=" + question + "'> "
            src_text +=   "Google</a> ?"

    # translate into target_language 
    if isinstance(target_language, str) and isinstance(src_text, str):
        if len(src_text) > 1:
            rs = ''
            if answer_in_format == 'html':
                # format the answer in HTML
                src_text = src_text.replace('[LEO_BOT]', '[LEO_BOT]<br/>')
                # convert the answer in markdown into html 
                # See https://www.devdungeon.com/content/convert-markdown-html-python
                rs_html = markdown.markdown(src_text, extensions=['fenced_code'])
                # translate into target language
                rs = translate_text(rs_html, target_language)
            else :
                src_text = src_text.replace('\n','<br/>')
                src_text = src_text.replace("```", "")                
                rs = translate_text(src_text, target_language)
                rs = format_string_for_md_slides(rs)
            return rs
        else:
            return src_text    
    elif src_text is None:
        return translate_text("Sorry, I can not answer your question !", target_language) 
    # done
    return str(src_text)

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
    