from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nexus_ai.sentence_sentiment_analysis.english import english_model
from nexus_ai.sentence_sentiment_analysis.arabic import arabic_model
from nexus_ai.utilities.util import process_google_reviews, remove_emoji, clean_review


def pred(reviews, source='Google Maps', sources=None):
    reviews = process_google_reviews(reviews, source=source, sources=sources)
    deleted_idx = []
    en_idx = []
    ar_idx = []
    en_reviews = []
    ar_reviews = []

    # remove emojis
    reviews = [remove_emoji(review) if review != None else None for review in reviews]
    # clean english reviews
    for i, review in enumerate(reviews):
        if review != None:
            try:
                if detect(review) == 'en':
                    reviews[i] = clean_review(review)
            # if the number of words in the text is shorter than 10 LangDetect would raise an exception
            # or if the text contain only numbers/symbols
            except LangDetectException:
                # if the review contain only numbers/symbols assign none to it for deleting later on
                reviews[i] = None

    # take english and arabic reviews and their original indexes
    for i, review in enumerate(reviews):
        # if the number of words in the text is shorter than 10 LangDetect would raise an exception
        if review != None:
            try:
                # take the emglish review and store it's original index for sorting later
                if detect(review) == 'en':
                    en_reviews.append(review) 
                    en_idx.append(i)
                # take the arabic review and store it's original index for sorting later
                elif detect(review) == 'ar':
                    ar_reviews.append(review)
                    ar_idx.append(i)
                # if the review is neither english or arabic assign none to it for deleting later on
                else: 
                    reviews[i] = None
            # if the text is short LangDetect would raise an exception
            # or if the text contain only numbers/symbols
            except LangDetectException:
                # if the review contain only numbers/symbols assign none to it for deleting later on
                reviews[i] = None

    # update deleted indexes list
    for i, review in enumerate(reviews):
        if review == None or review == '':
            deleted_idx.append(i)

    en_pred = english_model.pred(en_reviews)
    # combine the predictions with the original indexes 
    en_pred = list(zip(en_idx, en_pred))

    ar_pred = arabic_model.pred(ar_reviews)
    # combine the predictions with the original indexes 
    ar_pred = list(zip(ar_idx, ar_pred))

    # combine predictions
    en_pred.extend(ar_pred)
    predictions = en_pred
    # sort the reviews based on the original index 
    predictions = sorted(predictions, key= lambda tup: tup[0])

    # drop the indexes
    predictions = [tup[1] for tup in predictions]


    return deleted_idx, predictions




    
            