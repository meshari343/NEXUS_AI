from Reviews_Scrapper.Scrapper.FourSquare import foursquare_scrapper as fsqr
from Reviews_Scrapper.Scrapper.GoogleMaps import googlemaps_scrapper as googlemaps
from Reviews_Scrapper.Scrapper.TripAdvisor import tripadvisor_scrapper as tripadvisor

class Scrapper:
  
  def __init__(self,PATH):
    self.PATH = PATH;
      
  def read_foursquare_tips(self,URL,headless=False):
    return fsqr.read_tips(URL,self.PATH,)


  def read_google_maps_reviews(self,URL,headless=False):
    return googlemaps.read_reviews(URL,self.PATH,headless)

  def read_tripadvisor_reviews(self,URL,headless=False):
    return tripadvisor.read_reviews(URL,self.PATH,headless)