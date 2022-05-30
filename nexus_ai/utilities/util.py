from string import punctuation
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               u"\u23F0"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def clean_review(review, transform_punct=True, remove_punct=False, stopwords_=False, stemm=False):

    stop_words = stopwords.words('english')
    ps = PorterStemmer()
    token_dict = {'.': '||*Period*||', ',': '||*Comma*||', '"': '||*Quotation_Mark*||', 
                    ';': '||*Semicolon*||', '!': '||*Exclamation_mark*||', '?': '||*Question_mark*||',
                    '(': '||*Left_Parentheses*||', ')':'||*Right_Parentheses*||', '-': '||*Dash*||', '\n': '||*Return*||'}

    # lower case all charcters to normalize them 
    review = review.lower()
    # clearing stop words for example (a, we, on)
    # steemming the words(returning them to their root for example (stopped --> stop)
    if stopwords_ and stemm:
        review = ' '.join([ps.stem(word) for word in review.split() if not word in stop_words])
    elif stopwords_:
        review = ' '.join([word for word in review.split() if not word in stop_words])           
    elif stemm:
        review = ' '.join([ps.stem(word) for word in review.split()])

    # replace punctuation with token_dict values speceifed above 
    # while putting space as to not create multiple instance of the same word 
    # for example eating. would be treated as a new word diffrent from eating
    if transform_punct:
        for key, token in token_dict.items():
            review = review.replace(key, ' {} '.format(token))
    elif remove_punct:
        review = ''.join([c for c in review if c not in punctuation])

    # remove extra whitespaces
    review = ' '.join(review.split())

    return review

def process_text(text, only_english=False):
    if '(Translated by Google)' in text and '(Original)' in text:
        text  = text.replace('(Translated by Google)', '').partition('(Original)')
        try:
            # take the original review if it's non-empty or arabic, else take the translation if it's english
            if not only_english:
                text = text[2] if text[2]!='' and detect(text[2]) == 'ar' else(text[0] if detect(text[0]) == 'en' else None)
            else:
                text = text[0] if text[0]!='' and detect(text[0]) == 'en' else None
        except LangDetectException:
            text = None
    return text

def process_google_reviews(reviews, source, sources, only_english=False):
    # proccing google reviews if the list of each review source is provided
    if sources:
        # process reviews coming from google maps 
        reviews = [process_text(reviews[i], only_english=only_english) if sources[i] == 'Google Maps' else reviews[i] for i in range(len(reviews))]

    # if no specific source for each review is provided 
    # and if the overall source is google maps process all reviews
    elif source == 'Google Maps':    
        reviews = [process_text(review, only_english=only_english) for review in reviews]


    return reviews


