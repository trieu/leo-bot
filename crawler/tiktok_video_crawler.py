
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import constant
import requests

crawler_driver = False
def get_web_driver():
    global crawler_driver
    if not crawler_driver:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--ignore-certificate-errors')
        # options.add_argument('--window-size=1280,900')
        options.path = constant.CHROME_DRIVER_PATH
        options.page_load_strategy = 'eager'
        crawler_driver = webdriver.Chrome(options)
    return crawler_driver

def convert_string_to_number(input_str):
    s = re.sub(r'[^0-9][K][M]', '', input_str.upper())
    # Try to convert the string to an integer
    number = 0
    try:
        s = s.strip()  # Remove leading/trailing spaces
        if s[-1] == 'K':
            multiplier = 10 ** 3  # 1,000
            numeric_part = float(s[:-1])  # Remove the 'K' and convert the rest to a float
            number = int(numeric_part * multiplier)
        elif s[-1] == 'M':
            multiplier = 10 ** 6  # 1,000,000
            numeric_part = float(s[:-1])  # Remove the 'M' and convert the rest to a float
            number = int(numeric_part * multiplier)
        else:
            # No letter suffix, assume it's already a number
            number = float(s)
        # If it fails, try to convert the string to a float
    except ValueError:
        try:
            number = float(s)
        # If it fails, return None
        except ValueError:
            return None
    # Return the number
    return number

def get_html_from_url_by_chrome(url, elementSelector):
    try: 
        driver = get_web_driver()
        driver.get(url)
        delay = 20 # seconds
        try:
            data_node = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CSS_SELECTOR, elementSelector)))
            if data_node is not None:
                print("Found element " + elementSelector)
            driver.execute_script("window.stop()")
        except TimeoutException:
            print("Loading took too much time!")
        html = driver.page_source
        return html
    except: 
        print('Error when requests.get url: ' + url)
    return ""

def get_html_from_url_by_request(url):
    try: 
        agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        headers = { 'User-Agent': agent }
        r = requests.get(url, headers=headers)
        return r.text
    except: 
        print('Error when requests.get url: ' + url)
    return ""

def get_video_view(url:str, selector:str, use_chrome:bool = False):
    print('---------------------------------------')
    print('get_video_view url: ' + url)
    if use_chrome :
        html = get_html_from_url_by_chrome(url, selector)
    else: 
        html = get_html_from_url_by_request(url)
    if len(html) > 0 :
        soup = BeautifulSoup(html, "html5lib")
        title_node = soup.select_one('title')
        print("---------------------------")
        print(title_node.text)
        price_node = soup.select_one(selector)
        if price_node is not None:
            price_str = price_node.text
            price = convert_string_to_number(price_str)
            return price
    return -1


######################################################

selectorLikeCount = 'strong[data-e2e="like-count"]'

url1 = 'https://www.tiktok.com/@clap_h/video/7282667642584861998'
view1 = get_video_view(url1, selectorLikeCount, True)
print('Video Like: ', view1)

url2 = 'https://www.tiktok.com/@1longviboss/video/7517199612936015111'
view2 = get_video_view(url2, selectorLikeCount, True)
print('Video Like: ', view2)





