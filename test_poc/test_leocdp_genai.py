#### https://medium.com/@scholarly360/mistral-7b-complete-guide-on-colab-129fa5e9a04d

# Local AI LLM Model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from google.cloud import translate_v2 as translate

# Try PALM from Google AI
import google.generativeai as palm
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
palm.configure(api_key=GOOGLE_GENAI_API_KEY)



def translate_text(target: str, text: str) -> dict:
    """Translates text into the target language.
    """
    translate_client = translate.Client()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    result = translate_client.translate(text, target_language=target)
    return result['translatedText']


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

    pipe = pipeline(
            "text-generation",
            model=model_4bit,
            tokenizer=tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=512,
            do_sample=True,        
            temperature=0.68,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
    )
    return pipe

# load pipeline

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
pipe_text_generation = load_llm_pipeline()
llm_model = HuggingFacePipeline(pipeline=pipe_text_generation)
torch.cuda.empty_cache()

def start_leo_bot(question: str):
    context = """ In any knowledge domain, """
    template = """<s> [INST] Your name is LEO and you are the AI bot is created by Trieu at trieu@leocdp.com. 
    The answer should be clear from the context :
    {context} {question} [/INST] </s>
    """
    prompt = PromptTemplate(template=template, input_variables=["question","context"])

    # set pipeline into LLMChain with prompt and llm model  
    llm_chain = LLMChain(prompt=prompt, llm=llm_model)
    response = llm_chain.run({"question":question,"context":context})

    src_text = f"'{response}'".strip() 
    if src_text == "":
        src_text = palm.generate_text(prompt=question).result
    # print or save into database
    # print(src_text)    
    trs_text = translate_text('vi',src_text)
    print(trs_text)

def start_main_loop():
    #### Prompt in loop
    while True:
        question = input("\n [Please enter a question] \n").strip()
        if question == "exit" or len(question) == 0:
            # done
            break
        else:
            q = translate_text('en',question) 
            start_leo_bot(q)

# start local bot in command line
start_main_loop()