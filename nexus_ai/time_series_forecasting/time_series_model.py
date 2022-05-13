import pandas as pd
import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import warnings
from pmdarima import  auto_arima

# subtacting months from today date and returning a str with the year and month e.g.2022/04
def subtract_months(months):
    return (date.today() - relativedelta(months=months)).strftime('%Y/%m')


# subtacting weeks from today date and returning a str with the year and month e.g.2022/04
def subtract_weeks(weeks):
    return (date.today() - timedelta(weeks=weeks)).strftime('%Y/%m')


# subtacting days from today date and returning a str with the year and month e.g.2022/04
def subtract_days(days):
    return (date.today() - timedelta(days=days)).strftime('%Y/%m')


# a dictionary to map the date texts to a str with the month and year e.g. a month ago to 2022/04
def get_dic():
    return {
        'a month ago': subtract_months(1),
        '2 months ago': subtract_months(2),
        '3 months ago': subtract_months(3),
        '4 months ago': subtract_months(4),
        '5 months ago': subtract_months(5),
        '6 months ago': subtract_months(6),
        '7 months ago': subtract_months(7),
        '8 months ago': subtract_months(8),
        '9 months ago': subtract_months(9),
        '10 months ago': subtract_months(10),
        '11 months ago': subtract_months(11),
        'a week ago': subtract_weeks(1),
        '2 weeks ago': subtract_weeks(2),
        '3 weeks ago': subtract_weeks(3),
        '4 weeks ago': subtract_weeks(4),
        'a day ago': subtract_days(1),
        '2 days ago': subtract_days(2),
        '3 days ago': subtract_days(3),
        '4 days ago': subtract_days(4),
        '5 days ago': subtract_days(5),
        '6 days ago': subtract_days(6),
    }

def pred(json): 
    text_to_date = get_dic()
    df = pd.read_json(json)
    df = df[df['date'].isin(text_to_date.keys())]
    df["date"].replace(text_to_date, inplace=True)
    df = df.groupby(['date']).mean()
    
    warnings.filterwarnings("ignore")
    stepwise_fit = auto_arima(
        df.values,
        start_p=0,
        start_q=0,
        max_p=6,
        max_q=6,
        start_P=0,
        start_Q=0,
        max_P=6,
        max_Q=6,
        random_state=12,
    #     d = 2,
    #     stationary=True, 
    #     seasonal=True, 
    #     m=2,
        stepwise=True  
    )

    stepwise_fit.summary()
    
    time_series = df.values
    forecasting = np.array(stepwise_fit.predict(3)).clip(5).reshape(-1, 1)

    np.concatenate((time_series, forecasting))
    
    return time_series.tolist()   
