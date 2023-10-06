# leo-bot

- This bot works a proxy with the backend of Chat GPT and Google Generative AI
- For the Chat GPT, you need OpenAI key (https://platform.openai.com) 
- For the Google Generative AI, please check https://developers.generativeai.google/guide 
- Author: https://github.com/trieu

## In Ubuntu server, follow this checklist to run LEO BOT

1. Need Python 3.10
2. sudo apt install python-is-python3
3. curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10
4. pip install virtualenv (need to refresh the session of login shell)
5. Need to create a file .env to store environment variables
6. In the file .env, set value for OPENAI_API_KEY, GOOGLE_GENAI_API_KEY, REDIS_USER_SESSION_HOST and REDIS_USER_SESSION_PORT
7. Set correct DIR_PATH in start_app.sh
8. Run ./start_app.sh
9. The app will be start at the host 0.0.0.0 with port 8888

## More document links

* https://python.langchain.com/en/latest/getting_started/getting_started.html
* https://platform.openai.com/account/api-keys