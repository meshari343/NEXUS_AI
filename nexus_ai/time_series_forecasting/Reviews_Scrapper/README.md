## Requirements:

- chromedriver should be Installed on your machine
- installing requirements.txt packages

## USAGE

```
import Scrapper.myScrapper as myScrapper

## chromedriver PATH
PATH  =  'C:\Program Files (x86)\chromedriver.exe'


### IMPORTANT
scrapper  =  myScrapper.Scrapper(PATH)


# FourSquare
data  =  scrapper.read_foursquare_tips(URL='SOME URL')

# TripAdvisor
data  =  scrapper.read_tripadvisor_reviews(URL='SOME URL')

#Google Maps
data  =  scrapper.read_google_maps_reviews(URL='SOME URL')
```
