def clean_reviews(reviews_array):
  
    for review in reviews_array:
        old_val = review['rating'][1]
        newVal = float(old_val.split('_')[1])/10
        review['rating'] = newVal

    return reviews_array

def clean_scraped_data(place):
    cleaned_data = []
    count_nulls = 0
    count_not_nulls = 0
    
    if place != None:
        print(place['place_rating'])
        place['place_rating']= float( place['place_rating'].split()[0] )
        clean_reviews(place['reviews'])
        cleaned_data.append(place)
        count_not_nulls += 1  
    else:
        count_nulls += 1  
    
    # print('nulls: ',count_nulls)
    # print('not nulls: ',count_not_nulls)

    return cleaned_data