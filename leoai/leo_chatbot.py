from datetime import date

# Local AI LLM Model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# need Google translate to convert input into English
from google.cloud import translate_v2 as translate

# Try PALM from Google AI
import markdown
import google.generativeai as palm
import os
from dotenv import load_dotenv
load_dotenv()

# to use local model "Mistral-7B", export LEOAI_LOCAL_MODEL=true
LEOAI_LOCAL_MODEL = os.getenv("LEOAI_LOCAL_MODEL") == "true"
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
TEMPERATURE_SCORE = 0.69

# init PaLM client as backup AI
palm.configure(api_key=GOOGLE_GENAI_API_KEY)

# Translates text into the target language.
def translate_text(target: str, text: str) -> dict:
    if text == "" or text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    result = translate.Client().translate(text, target_language=target)
    return result['translatedText']

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
    template = """<s> [INST] Your name is LEO_BOT and you are the AI bot is created by Mr.Triều at LEOCDP.com. 
    The answer should be clear from the context :
    {context} {question} [/INST] </s>
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
            src_text = palm.generate_text(prompt=prompt_text, temperature=temperature_score).result    
        except Exception as error:
            print("An exception occurred:", error)
        
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
                rs = translate_text(target_language, rs_html)
            else :
                rs = translate_text(target_language, src_text)
            return rs
        else:
            return src_text    
    elif src_text is None:
        return translate_text(target_language, "Sorry, I can not answer your question !") 
        # no need to translate
    return str(src_text)