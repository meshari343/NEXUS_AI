# NEXUS AI

nexus analytics is a platform powered by AI aimed to help
business owners in Restaurant and Café sector to make data-driven
decisions based on their customer's reviews on social
media platforms.

in this repository you would find the ML models that support this platform, you can view the platform that was implemented using Node.js in this [repository](https://github.com/SimplyRayan/Nexus-Backend). 

## setup

to use the API, first of all make sure you are using python 3.10 or create a conda virtual enviroment with this python version, and then run the following commands in the terminal to clone the repository and install the dependencies.
### Clone the repository
```bash
git clone https://github.com/meshari343/NEXUS_AI
```
after cloning the repository, download the [ABSA model](https://drive.google.com/file/d/1uSpLTYWCDUMujGy-NqDu-nPhpyzZGRwv/view?usp=sharing) and unpack it to (nexus_ai/ABSA/ATEPC_models)
### (Optional) Linux creating virtual environment 
```bash
pip instal venv
python -m venv venv                          # create venv virtual environment
source venv/bin/activate                     # activate venv virtual environment 
```
### (Optional) Windows creating virtual environment 
```bash
pip instal venv
python -m venv venv                          # create venv virtual environment
venv/source/activate                         # activate venv virtual environment 
```
### install the package & dependencies
```bash
pip install -r requirements.txt              # Install the dependencies
cd NEXUS_AI
pip install .                                # install nexus_ai package
python -m spacy download en_core_web_sm      # download spacy model
```
### Run the API
```bash
uvicorn nexus_ai.fastapi.main:app            # add --reload at the end if you would like to modify the code
```
after that, you can go to http://127.0.0.1:8000 in your local browser, to see the API documentation.
### Data sample
below is a JSON sample input dataset, to use in testing out the API models.
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

if you would like to use the model inside python, instead of making use of the API, do the same setup steps above, other than the last one for runnig the API, and then in python, you can use the models like this.
### sentence sentiment analysis
```python
from nexus_ai.sentence_sentiment_analysis import sentence_sentiment_analysis_model
deleted_idx, predictions = sentence_sentiment_analysis_model.pred(reviews)
```
* reviews: List of sentences to classify.
* predictions: The prediction for each sentence.
* deleted_idx: zero-length reviews/non-English reviews are gonna be deleted, and the deleted indexes are returned.



### aspect based sentiment analysis (ABSA)
```python
from nexus_ai.ABSA import ABSA_model
deleted_idx, df_predictions = ABSA_model.pred(reviews)
```
- reviews: List of sentences to classify.
- df_predictions: Dictionary with three keys: 
    - "aspects": List of lists, each list contains the aspects for each review
    - "aspects_sentiment": List of lists, each list contains the polarity for each aspect
    - "aspects_description": List of lists, each list contains the description for each aspect



### time series forecasting 
```python
from nexus_ai.time_series_forecasting import time_series_model
time_series = time_series_model.pred(data, seasonal=True)
```
* data: JSON string with the same format as the json sample above, but the username, and text are not optional.
* seasonal: Optional boolean parameter, to decide whetever to use seasonal_test, note that without seasonal test the model building process is x6 times faster, but may give lower results.
* time_series: a dictionary containing four keys:
    * "past_values": the last year monthly avarage ratings, min: 3 months, max: 12 months.
    * "past_dates": the year/month associated with each value of the past
    * "future_forecasting": the prediction for the next 3 months avarage ratings.
    * "future_dates": the year/month associated for each value of the forecasted future.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.