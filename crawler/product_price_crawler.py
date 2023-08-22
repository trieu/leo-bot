
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
    num_str = re.sub(r'[^0-9]', '', input_str)
    
    # Try to convert the string to an integer
    try:
        number = int(num_str)
    # If it fails, try to convert the string to a float
    except ValueError:
        try:
            number = float(num_str)
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

def get_product_price(url:str, selector:str, use_chrome:bool = False):
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

url1 = 'https://www.fahasa.com/30-giay-khoa-hoc-30-giay-khoa-hoc-du-lieu.html'
selector1 = '#catalog-product-details-price .price'
price1 = get_product_price(url1, selector1)
print('Fahasa Product Price: ', price1)

url2 = 'https://www.lazada.vn/products/xe-may-honda-future-125-fi-cao-cap-2023-i1599437312-s6862243432.html'
selector2 = "#module_product_price_1 > div > div > span"
price2 = get_product_price(url2, selector2, True)
print('Lazada Product Price: ', price2)

# url3 = 'https://shopee.vn/Xe-m%C3%A1y-Honda-Vision-2023_Phi%C3%AAn-b%E1%BA%A3n-Th%E1%BB%83-thao-i.313043031.23913942525?sp_atk=452f76e2-4d42-47a3-bc34-28f0c1c67c9a&xptdk=452f76e2-4d42-47a3-bc34-28f0c1c67c9a'
# selector3 = "div.pqTWkA"
# price3 = get_product_price(url3, selector3)
# print('Lazada Product Price: ', price3)



