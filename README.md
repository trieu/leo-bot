# LEO CHATBOT for LEO CDP and the World (leo-bot)

- The LEO BOT works an AI chatbot with the backend using Google Generative AI (PaLM 2) and Mistral-7B
- For the Google Generative AI, please check more details at https://developers.generativeai.google/guide 
- Author: Trieu Nguyen at https://LEOCDP.com (my Github https://github.com/trieu)

## In Ubuntu server, follow this checklist to run LEO BOT

1. Need Python 3.10, run following commands
```
sudo apt install python-is-python3
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10
pip install virtualenv
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
2. You need to refresh the session of login shell after install python-is-python3
3. Need to create a file .env to store environment variables
4. In the file .env, set value like this example
```
    LEOBOT_DEV_MODE=true
    HOSTNAME=leobot.example.com
    LEOAI_LOCAL_MODEL=false
    GOOGLE_GENAI_API_KEY=
    REDIS_USER_SESSION_HOST=127.0.0.1
    REDIS_USER_SESSION_PORT=6480
```
5. Set correct DIR_PATH in start_app.sh, an example like this
```
DIR_PATH="/build/leo-bot/"
```
6. Run ./start_app.sh
7. The LEO BOT is started at the host 0.0.0.0 with port 8888
8. For demo and local test, open redis-cli, run: hset demo userlogin demo
9. Go to the HOSTNAME to test (e.g: https://leobot.example.com )