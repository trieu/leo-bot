#### https://medium.com/@scholarly360/mistral-7b-complete-guide-on-colab-129fa5e9a04d

# importing module
#from importlib.machinery import SourceFileLoader
#leo_chatbot = SourceFileLoader("leoai","leoai/leo_chatbot.py").load_module()

from leoai import leo_chatbot

def start_main_loop():
    while True:
        question = input("\n [Please enter a question] \n").strip()
        if question == "exit" or len(question) == 0:
            # done
            break
        else:
            q = leo_chatbot.translate_text('en',question) 
            print(leo_chatbot.ask_question('vi', q))

# start local bot in command line
start_main_loop()