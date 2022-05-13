from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time


from Reviews_Scrapper.Scrapper.TripAdvisor import clean_tripadvisor as clean_tripadvisor
# import Scrapper.TripAdvisor.clean_tripadvisor as clean_tripadvisor

#########################################
PLACE_NAME_CLASSNAME = '_3a1XQ88S'
PLACE_RATING_CLASSNAME = 'zWXXYhVR'
REVIEW_BOX_CLASSNAME = 'reviewSelector'
REVIEW_RATING_CLASSNAME = 'ui_bubble_rating'
REVIEW_HEADER_CLASSNAME = 'noQuotes'
REVIEW_DATE_CLASSNAME= 'ratingDate'
REVIEW_USERNAME_CLASSNAME= 'info_text'

##########################################
def collect_reviews(URL,PATH,headless=False):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')   

    driver = webdriver.Chrome(PATH,options=options)
    
    driver.set_window_size(1920, 1080)
    driver.get(URL)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'REVIEWS')))
    
    html = driver.page_source
    soup = BeautifulSoup(html,'lxml')
    place_name= soup.find(class_=PLACE_NAME_CLASSNAME).text
    place_rating= soup.find(class_=PLACE_RATING_CLASSNAME).get('title')

    reviews = []

    try:
        while True:
            html = driver.page_source
            soup = BeautifulSoup(html,'lxml')
            reviews_list = soup.find_all(class_='reviewSelector')

            for review_html in reviews_list:
                # print(review_html)
                review_data ={
                    'rating':review_html.find(class_='ui_bubble_rating').get('class'),
                    'header':review_html.find(class_='noQuotes').text,
                    'text': review_html.find('p').text,
                    'date': review_html.find(class_=REVIEW_DATE_CLASSNAME).get('title'),
                    'username' :review_html.find(class_=REVIEW_USERNAME_CLASSNAME).find_all('div')[0].text
                }
                reviews.append(review_data)

            print(reviews)
            nextBtn = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'next')))
            print(nextBtn)
            nextBtn.click()
            time.sleep(3)
                
    except Exception:
        pass

    data ={
        'place_name':place_name,
        'place_rating':place_rating,
        'reviews':reviews
    }
    driver.quit()
    return data



def read_reviews(URL,PATH,headless=False):
    if PATH == None:
        return print('ERROR: PATH WAS NOT SET')
    
    data = collect_reviews(URL,PATH,headless)
    data = clean_tripadvisor.clean_scraped_data(data)

    return data
    