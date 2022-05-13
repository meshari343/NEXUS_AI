# NEXUS AI

nexus analytics is a platform powered by AI aimed to help
business owners in Restaurant and Café sector to make data-driven
decisions based on their customer's reviews on social
media platforms.

in this repository you would find the ML models that support this platform, you can view the platform that was implemented using Node.js in this [repository](https://github.com/SimplyRayan/Nexus-Backend). 

## setup

to use the API, Run the following commands in the terminal to clone the repository and install the dependencies.

after running the first command and cloning the repository, download the [ABSA model](https://drive.google.com/file/d/1uSpLTYWCDUMujGy-NqDu-nPhpyzZGRwv/view?usp=sharing) and unpack it to (nexus_ai/ABSA/ATEPC_models)
```bash
git clone https://github.com/meshari343/NEXUS_AI
cd MLND_capstone
# the below commented commands is for creating the virtual environment and they are optional:
# pip instal venv
# python -m venv venv                          # create venv virtual environment
# source venv/bin/activate                     # activate venv virtual environment for (linux)
# venv/source/activate                         # activate venv virtual environment for (windows)
pip install -r requirements.txt                # Install the dependencies
pip install -e                                 # install nexus_ai package
uvicorn nexus_ai.fastapi.main:app --reload     # run API
```
after that, you can go to http://127.0.0.1:8000 in your local browser.

below is a JSON sample input dataset, to use in testing out the API.
```json
[
{"rating": 5.0,
 "date": "3 months ago",
 "text": "Best coffee place in Riyadh, really liked their coffee, their staff, the atmosphere is quite good to!",
 "username": "Anonymous"},
 {"rating": 4.0,
 "date": "a year ago",
 "text": "Love the name and the place. Friendly staff and nice coffee. Good place to study and work. There is wifi. I liked the colors of the ceramic cups but they should get the perfect size for the flat white. They served my flat white in a large cup \u201c latte cup\u201d",
 "username": "Wafaa Alhusain"},
 {"rating": 5.0,
 "date": "9 months ago",
 "text": "It\u2019s so quiet and perfect for studying or working here. The staff is friendly and very helpful. Try their honey cake & caramel bites, sooo good!.",
 "username": "Bashaer Aljabr"},
 {"rating": 4.0,
 "date": "5 months ago",
 "text": "Good for studying. I\u2019ve tried iced peanut latte. It was good but there is no taste of peanut butter. They have free internet access. They don\u2019t have almond milk. The sweets are below average. You can book a meeting room.",
 "username": "B"},
 {"rating": 5.0,
 "date": "a year ago",
 "text": "Amazing coffee shop, cozy place , delicious coffee. They have workstations and meeting rooms, the place just speaks productivity, great for doing work or studying ! Definitely coming back .",
 "username": "Hussam Ahmed"}]
```

## Usage

if you would like to use the model inside python, instead of making use of the API, do the same setup steps above, other than the last one for runnig the API, and then in python, you can use the models like this
```python
# 1) sentence sentiment analysis
from nexus_ai.sentence_sentiment_analysis import sentence_sentiment_analysis_model
# reviews: a list of reviews
# predictions: the prediction for each review 
# deleted_idx: zero-length reviews/non-English reviews are gonna be deleted 
# and the deleted indexes are returned 
deleted_idx, predictions = sentence_sentiment_analysis_model.pred(reviews)



# 2) aspect based sentiment analysis (ABSA)
from nexus_ai.ABSA import ABSA_model
# reviews: a list of reviews
# df_predictions: a dataframe with three columns: 
# 1-aspects in each review 2-polarity for each aspect 3-description for each aspect
deleted_idx, df_predictions = ABSA_model.pred(reviews)



# 3) time series forecasting 
from nexus_ai.time_series_forecasting import time_series_model
# data: JSON string
# time_series: a list with the last year monthly avarage ratings
# plus the prediction of the avarage ratings for the next three months
time_series = time_series_model.pred(data)
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.