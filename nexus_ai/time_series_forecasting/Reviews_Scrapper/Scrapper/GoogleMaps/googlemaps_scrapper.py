from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


from bs4 import BeautifulSoup
import pickle

import time
import sys
import re

###############################################

# PATH = None;

# def set_driver_PATH(path):
#   PATH = path

###############################################

REVIEW_CONTAINER_CLASS_NAME = 'jxjCjc'
RATING_CLASS_NAME = 'Fam1ne'
DATE_CLASS_NAME = 'dehysf'
TEXT_CLASS_NAME = 'Jtu6Td'
TEXT_CLASS_NAME = 'Jtu6Td'
TEXT_CLASS_NAME_FULL_TEXT = 'review-full-text' # MAY BE THERE MAY BE NOT
USERNAME_CLASSNAME = 'TSUbDb'



def extract_reviews(html):
    soup = BeautifulSoup(html,'lxml')
    reviews =  soup.find_all(class_ = REVIEW_CONTAINER_CLASS_NAME)
    data = []

    for review_html in reviews:
        review_text = review_html.find(class_= TEXT_CLASS_NAME_FULL_TEXT)
        if review_text == None:
            review_text = review_html.find(class_= TEXT_CLASS_NAME)
        review_data ={
                    'rating':float( review_html.find(class_= RATING_CLASS_NAME).get('aria-label').split()[1] ),
                    'date':review_html.find(class_= DATE_CLASS_NAME).text,
                    'text':review_text.text,
                    'username':review_html.find(class_= USERNAME_CLASSNAME).text
                }
        
        data.append(review_data)

    return data


###############################################

###############################################


SCROLLABLE_ELEMNT_CLASSNAME = 'review-dialog-list'

# it reads the reviews and returns the html of the page
def collect_reviews(url,PATH,headless=False):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(PATH,options=options)

    driver.set_window_size(1920, 1080)
    html = None
    try:
        
        driver.get('https://www.google.com')
        time.sleep(1.5)
        ## load previous cookies so that the language becomse english instead of location language
        cookies = pickle.load(open("Reviews_Scrapper/Scrapper/GoogleMaps/my_cookies.pkl", "rb")) 
        for cookie in cookies:
            driver.add_cookie(cookie)
        
        driver.get(url)
        time.sleep(3)


        reviews_count = get_current_reviews_count(driver)
        scrolled_reviews = 0
      
        while scrolled_reviews < reviews_count:
            scrollable_element = driver.find_element_by_class_name(SCROLLABLE_ELEMNT_CLASSNAME)
            driver.execute_script(
                    'arguments[0].scrollTop = arguments[0].scrollHeight', 
                        scrollable_element
                    )
            
            scrolled_reviews+=  REVIEWS_PER_PAGE
            time.sleep(1.5)
            
        html = driver.page_source
    
    except Exception:
        print('ERROR !')
        print(sys.exc_info())

    finally:
        driver.quit()
        

    return html






REVIEWS_PER_PAGE = 10
REVIEWS_COUNTER_CLASSNAME= 'z5jxId'

## this method is used to read number reviews in the current page, it helps in knowing when to stop scrolling
def get_current_reviews_count(driver):
    try:
        counter_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, REVIEWS_COUNTER_CLASSNAME))
        )
        REVIEWS_COUNT = driver.execute_script("return arguments[0].textContent",counter_element)
        REVIEWS_COUNT = REVIEWS_COUNT.split(' ')[0]
        # REVIEWS_COUNT = unidecode(REVIEWS_COUNT)
        
        REVIEWS_COUNT = re.sub(',(?!\s+\d$)', '', REVIEWS_COUNT) 
        return int(REVIEWS_COUNT)
    except:
        print('ERROR WHILE COUNTING REVIEWS')
        print(sys.exc_info())
        return False
    



## takes a url of the reviews page and returns the reviews data
def read_reviews(URL,PATH,headless=False):
    print('Google Maps Scrapper')
    html =  collect_reviews(URL,PATH,headless)
    data = extract_reviews(html)
    return data
    
