# NEXUS AI

nexus analytics is a platform powered by AI aimed to help
business owners in Restaurant and Café sector to make data-driven
decisions based on their customer's reviews on social
media platforms.

in this repository you would find the API to the ML models that support this platform, you can view the platform that was implemented using Node.js in this [repository](https://github.com/SimplyRayan/Nexus-Analytics). 

## setup

### using docker 
#### Clone the repository
```bash
git clone https://github.com/meshari343/NEXUS_AI
```
**important:** after cloning the repository, download the [ABSA model](https://drive.google.com/file/d/1uSpLTYWCDUMujGy-NqDu-nPhpyzZGRwv/view?usp=sharing) and unpack it to (nexus_ai/ABSA/ATEPC_models).


You should also download and unpack [arabic sentence analysis model](https://drive.google.com/file/d/1nezfOeqGvbC-R9QniZxeRp2iy6YQP-pi/view?usp=sharing) to (nexus_ai/sentence_sentiment_analysis/arabic/models).
#### create docker container
```bash
CD NEXUS_AI
docker build -t nexus_ai .
docker run --name nexus_ai -p 8000:8000 -d nexus_ai
```
after that, you can go to http://127.0.0.1:8000 in your local browser, to see the API documentation.
### using traditional method
make sure you are using python 3.9 or create a virtual environment using virtualenv with this python version, and then run the following commands in the terminal to clone the repository and install the dependencies.
#### Clone the repository
```bash
git clone https://github.com/meshari343/NEXUS_AI
```
**important:** after cloning the repository, download the [ABSA model](https://drive.google.com/file/d/1uSpLTYWCDUMujGy-NqDu-nPhpyzZGRwv/view?usp=sharing) and unpack it to (nexus_ai/ABSA/ATEPC_models).


You should also download and unpack [arabic sentence analysis model](https://drive.google.com/file/d/1nezfOeqGvbC-R9QniZxeRp2iy6YQP-pi/view?usp=sharing) to (nexus_ai/sentence_sentiment_analysis/arabic/models).
#### (Optional) Linux creating virtual environment 
```bash
pip instal venv
python -m venv venv                          # create venv virtual environment
source venv/bin/activate                     # activate venv virtual environment 
```
#### (Optional) Windows creating virtual environment 
```bash
pip instal venv
python -m venv venv                          # create venv virtual environment
venv/source/activate                         # activate venv virtual environment 
```
#### install the package & dependencies
```bash
pip install -r requirements.txt              # Install the dependencies
cd NEXUS_AI
pip install .                                # install nexus_ai package
python -m spacy download en_core_web_sm      # download spacy model
```
#### Run the API
```bash
python main.py                               
```
after that, you can go to http://127.0.0.1:8000 in your local browser, to see the API documentation.
### Data sample
below is a JSON sample input dataset to test out the API models.
```json
[
{"source": "Google Maps",
 "rating": 4.0,
 "date": "3 months ago",
 "text": "Me and my friends came here for our coffee and sweets after our hearty dinner. Thanks to Romnick for his coffee recommendation and the cheesecake.   Love every bit of drinks and desserts in this place. More power",
 "username": " Jackie A. M. "},
{"source": "Google Maps",
 "rating": 4.0,
 "date": "2 weeks ago",
 "text": "Saffron cake is decent, I like their design and the vibes around it",
 "username": " Sam B "},
 {"source": "Google Maps",
  "rating": 4.0,
  "date": "3 months ago",
  "text": "A place like home. Shine and bright ⭐⭐⭐🌃",
  "username": " I love riyadh "},
 {"source": "Google Maps",
  "rating": 5.0,
  "date": "a month ago",
  "text": "Great coffee and very welcoming employees especially Islam and chanja. Many thanks,,",
  "username": " naif "},
 {"source": "Google Maps",
  "rating": 1.0,
  "date": "3 months ago",
  "text": "I didn’t like their dulce de leche the syrup doesn’t taste like dulce de leche and has an odd taste to it almost like metal",
  "username": " Sweet Creature "},
]
```

## Usage

if you would like to use the model inside python, instead of making use of the API, do the same setup steps above, other than the last one for running the API, and then in python, you can use the models like this.
### sentence sentiment analysis
```python
from nexus_ai.sentence_sentiment_analysis import sentence_sentiment_analysis_model
deleted_idx, predictions = sentence_sentiment_analysis_model.pred(reviews, sources)
```
* reviews: List of sentences to classify.
* (Optional) sources: List with the source of each review, 'Google Maps' since their reviews needs additional preprocessing steps
* predictions: The prediction for each sentence.
* deleted_idx: zero-length reviews, non-Arabic reviews, and non-English reviews are gonna be deleted, and the deleted indexes are returned.



### aspect based sentiment analysis (ABSA)
```python
from nexus_ai.ABSA import ABSA_model
deleted_idx, predictions = ABSA_model.pred(reviews, sources)
```
- reviews: List of sentences to classify.
- (Optional) sources: List with the source of each review, 'Google Maps' since their reviews needs additional preprocessing steps
- predictions: Dictionary with three keys: 
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
    * "past_values": the last year monthly average ratings, min: 3 months, max: 12 months.
    * "past_dates": the year/month associated with each value of the past
    * "future_forecasting": the prediction for the next 3 months average ratings.
    * "future_dates": the year/month associated for each value of the forecasted future.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.