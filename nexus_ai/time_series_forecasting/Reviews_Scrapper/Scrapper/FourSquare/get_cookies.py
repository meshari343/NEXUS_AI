import pickle
import selenium.webdriver
import time
PATH = 'C:\Program Files (x86)\chromedriver.exe'
driver = selenium.webdriver.Chrome(PATH)
driver.get("http://www.google.com")
time.sleep(60)
pickle.dump( driver.get_cookies() , open("cookies.pkl","wb"))