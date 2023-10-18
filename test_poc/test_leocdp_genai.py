#### https://medium.com/@scholarly360/mistral-7b-complete-guide-on-colab-129fa5e9a04d

# Local AI LLM Model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

# need Google translate to convert input into English
from google.cloud import translate_v2 as translate

# Try PALM from Google AI
import google.generativeai as palm
import os
from dotenv import load_dotenv
load_dotenv()

# to use local model "Mistral-7B", export LEOAI_LOCAL_MODEL=true
LEOAI_LOCAL_MODEL = os.getenv("LEOAI_LOCAL_MODEL") == "true"
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

# init PaLM client as backup AI
palm.configure(api_key=GOOGLE_GENAI_API_KEY)

# Translates text into the target language.
def translate_text(target: str, text: str) -> dict:
    if text == "" or text is None:
        return ""
    translate_client = translate.Client()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    result = translate_client.translate(text, target_language=target)
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
            temperature=0.7,
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
    # clear GPU cache after init 
    torch.cuda.empty_cache()

# the main function to ask LEO
def ask_leo_assistant(target_language: str, question: str) -> str:
    context = """ In any knowledge domain, """
    template = """<s> [INST] Your name is LEO and you are the AI bot is created by Mr.Triều at LEOCDP.com. 
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
        src_text = palm.generate_text(prompt=prompt_text).result
    # translate into target_language 
    if isinstance(target_language,str):
        trs_text = translate_text(target_language, src_text)
        return trs_text
    else:
        return src_text

def start_main_loop():
    while True:
        question = input("\n [Please enter a question] \n").strip()
        if question == "exit" or len(question) == 0:
            # done
            break
        else:
            q = translate_text('en',question) 
            print(ask_leo_assistant('vi', q))

# start local bot in command line
start_main_loop()