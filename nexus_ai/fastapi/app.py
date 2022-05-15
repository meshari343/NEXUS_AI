import pandas as pd
from fastapi import FastAPI, status, Header
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, conlist, Extra
from typing import Optional
import json
# import pydantic
# from requests import Response
from nexus_ai.sentence_sentiment_analysis import sentence_sentiment_analysis_model
from nexus_ai.ABSA import ABSA_model
from nexus_ai.time_series_forecasting import time_series_model


class row(BaseModel):
    text: str
    date: str
    rating: int
    username: str
    class Config:
        extra = Extra.forbid


class data_model(BaseModel):
    __root__: conlist(row, min_items=1)


app = FastAPI()

@app.get("/")
async def root():
    return RedirectResponse(url='/docs')


@app.post("/time_series", status_code=status.HTTP_201_CREATED)
async def timeseries(data: data_model, seasonal: Optional[bool] = Header(default=False)):
    return time_series_model.pred(data.json(), seasonal)


@app.post("/sentiment_analysis", status_code=status.HTTP_201_CREATED)
async def sentence_sentiment_analysis(data: data_model):
    df = pd.read_json(data.json())
    reviews = list(df['text'])

    # deleted_idx: zero length reviews/non english reviews are gonna be deleted and the delted indexes are returned
    deleted_idx, predictions = sentence_sentiment_analysis_model.pred(reviews)
    df.drop(deleted_idx, axis=0, inplace=True)

    df['sentence_sentiment'] = predictions

    reviews = list(df['text'])
    # deleted_idx: zero-length reviews/non-English reviews are gonna be deleted and the deleted indexes are returned 
    deleted_idx, ABSA_predictions = ABSA_model.pred(reviews)
    df.drop(deleted_idx, axis=0, inplace=True)

    df['aspects'] = ABSA_predictions['aspect']
    df['aspects_sentiment'] = ABSA_predictions['sentiment']
    df['aspects_description'] = ABSA_predictions['description']

    json_str = df.to_json(orient='records')
    json_obj = json.loads(json_str)

    return json_obj



@app.post("/ABSA", status_code=status.HTTP_201_CREATED)
async def ABSA(data: data_model):
    df = pd.read_json(data.json())
    reviews = list(df['text'])

    # deleted_idx: zero-length reviews/non-English reviews are gonna be deleted and the deleted indexes are returned 
    deleted_idx, predictions = ABSA_model.pred(reviews)
    df.drop(deleted_idx, axis=0, inplace=True)

    df['aspects'] = predictions['aspect']
    df['sentiments'] = predictions['sentiment']

    json_str = df.to_json(orient='records')
    json_obj = json.loads(json_str)

    return json_obj


@app.post("/sentence_sentiment_analysis", status_code=status.HTTP_201_CREATED)
async def sentence_sentiment_analysis(data: data_model):
    df = pd.read_json(data.json())
    reviews = list(df['text'])

    # deleted_idx: zero-length reviews/non-English reviews are gonna be deleted and the deleted indexes are returned 
    deleted_idx, predictions = sentence_sentiment_analysis_model.pred(reviews)
    df.drop(deleted_idx, axis=0, inplace=True)

    df['prediction'] = predictions

    json_str = df.to_json(orient='records')
    json_obj = json.loads(json_str)

    return json_obj
