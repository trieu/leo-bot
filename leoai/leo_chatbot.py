
# Local AI LLM Model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
import datetime
from dotenv import load_dotenv
load_dotenv()

LEOAI_LOCAL_MODEL = os.getenv("LEOAI_LOCAL_MODEL") == "true"
TEMPERATURE_SCORE = 0.86

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
    if llm_model is not None:  
        llm_chain = LLMChain(prompt=prompt_tpl, llm=llm_model)
        response = llm_chain.run(prompt_data)

    src_text = f"{response}".strip()

    # done
    return str(src_text)
