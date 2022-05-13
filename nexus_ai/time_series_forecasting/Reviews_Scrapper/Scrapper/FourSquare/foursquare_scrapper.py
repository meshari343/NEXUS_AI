from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import selenium.common.exceptions
from bs4 import BeautifulSoup

import time
import pickle
######################## 


# USED TO Idetnify HTML Elements that of interest

TIP_LIST_ID = 'tipsList'; 
TIP_CLASSNAME = 'tip'
TIP_TEXT_CLASSNAME = 'tipText'
TIP_DATE_CLASSNAME = 'tipDate'
TIP_USERNAME = 'userName'
TIP_USER_VISITS = 'tipAuthorJustification'

PAGES_CONTAINER_CLASSNAME = 'paginationContainer'
PAGINATION_ELEMENTS_CLASSNAME = 'page'

#######################
def interceptor(request):
    del request.headers['User-Agent']  # Delete the header first
    request.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    request.headers['sec-fetch-site'] = 'same-origin'
    request.headers['sec-fetch-mode'] = 'cors'
    request.headers['sec-fetch-dest'] = 'empty'
    request.headers['sec-ch-ua-mobile'] = '?0'
    request.headers['referer'] = 'https://foursquare.com'
    request.headers['origin'] = 'https://foursquare.com'
    request.headers['sec-ch-ua'] = 'Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92'
    request.headers['accept-language'] = 'en-US,en;q=0.9'
    request.headers['accept-encoding'] = 'gzip, deflate, br'
    request.headers['accept'] = '*/*'



#######################
def get_tips_data(soup,current_page):
  tip_list = soup.find_all(class_=TIP_CLASSNAME)
  data = []
  for tip in tip_list:
    tip_data = {}
    tip_data['tip_text'] = tip.find(class_=TIP_TEXT_CLASSNAME).text
    tip_data['tip_date'] = tip.find(class_=TIP_DATE_CLASSNAME).text
    tip_data['tip_username'] = tip.find(class_=TIP_USERNAME).text
    # tip_data['tip_user_visits'] = tip.find(class_=TIP_USER_VISITS).text
    data.append(tip_data)

  hasNextPage= False;
  pages_container = soup.find(class_=PAGES_CONTAINER_CLASSNAME)
  if pages_container:
    pages_shown = pages_container.find_all(class_=PAGINATION_ELEMENTS_CLASSNAME)
    last_page_number = int(pages_shown[-1].text)
    if last_page_number-1 > current_page:
      print(f'LAST PAGE  = {last_page_number}, CURRENT PAGE = {current_page}',last_page_number > current_page)
      hasNextPage= True;
  return {
    'hasNextPage':hasNextPage,
    'data' :data
  }
      


def read_tips(url,PATH,headless=False):
    if PATH == None:
      return print('ERROR: PATH WAS NOT SET')

    options = webdriver.ChromeOptions()
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
    options.add_argument(f'user-agent={user_agent}')
    if headless:
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')   


    driver = webdriver.Chrome(PATH,options=options)
    driver.set_window_size(1920, 1080)
    driver.request_interceptor = interceptor
    # driver.get('https://foursquare.com/')
    # time.sleep(1.5)
    # ## load previous cookies so that the language becomse english instead of location language
    # cookies = pickle.load(open("FourSquare/cookies.pkl", "rb")) 
    # for cookie in cookies:
    #     driver.add_cookie(cookie)
    
    driver.get(url)
    

    

    driver.get(url)
    current_page = 0;
    data = [];
    
    while True:
      WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, TIP_LIST_ID)))

      time.sleep(1)
      html = driver.page_source
      soup = BeautifulSoup(html,'lxml')

      res = get_tips_data(soup,current_page)

      data += res['data']
      
      if res['hasNextPage']:
        current_page += 1
        nextPageBtn = driver.find_element_by_class_name(f'page{current_page}')
        driver.execute_script("arguments[0].scrollIntoView();",nextPageBtn)
        nextPageBtn.click()
        time.sleep(2)
      else: 
        break;

    driver.quit()
    print('TOTAL TIPS COLLECTED =',len(data))

    return data


