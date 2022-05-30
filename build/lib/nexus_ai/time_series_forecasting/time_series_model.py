import pandas as pd
import numpy as np
from datetime import date, timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
from pmdarima import  auto_arima

# subtacting months from today date and returning a str with the year and month e.g.2022/04
def subtract_months(months):
    return (date.today() - relativedelta(months=months)).strftime('%Y/%m')
    
# add months from given date and return a str with the year and month e.g.2022/04
def add_months(date_, months):
    return (date_ + relativedelta(months=months)).strftime('%Y/%m')

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
def complete_months(df):
    complete_year = []
    # appending today month and year e.g., 2022/05
    complete_year.append(date.today().strftime('%Y/%m'))
    # appending the past 11 months from the current month
    for i in range(11):
        complete_year.append(subtract_months(i+1))
    # create an array with nan values to be the dataframe values with the complete year being the index
    nan_array = np.empty((12, 1))
    nan_array[:] = np.nan
    # creating a dataframe with complete year as index and nan values
    df_complete_year = pd.DataFrame(nan_array, index=complete_year, columns=['rating'])
    # getting the missing months from the dataframe
    missing_months = df_complete_year.index.difference(df.index)
    # appending missing months to the dataframe
    df = pd.concat((df, df_complete_year.loc[missing_months, :]))
        
    return df
        
def pred(json, seasonal=False): 
    text_to_date = get_dic()
    df = pd.read_json(json)
    # remove dates outside of the dictionary e.g., a year ago
    df = df[df['date'].isin(text_to_date.keys())]

    # transform the dates from text to a text with the year and month e.g., a month ago --> 2022/04
    df["date"].replace(text_to_date, inplace=True)

    # getting the avarage rating for each month in the last year
    df = df.groupby(['date']).mean()

    # making sure there's at least 4 months with ratings
    if len(df) < 4:
        return None

    # compelte months with no rating, with nan values for now
    df = complete_months(df)

    # if there's months with no ratings at the start or end of the dataframe drop them
    first_idx = df.first_valid_index()
    last_idx = df.last_valid_index()
    df = df.loc[first_idx:last_idx]

    # filling missing values (months with no ratings) in the middle
    df.fillna(method='bfill', inplace=True)

    warnings.filterwarnings("ignore")
    if seasonal:
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
            seasonal=True, 
            m=4,
            D=None,
            seasonal_test='ch',
            stepwise=True 
        )
    else:
        stepwise_fit = auto_arima(
            df.values,
            start_p=0,
            start_q=0,
            max_p=6,
            max_q=6,
            random_state=12, 
            seasonal=False,
            stepwise=True,
        )
    
    time_series = df.values.flatten().tolist()
    forecasting = np.array(stepwise_fit.predict(3)).clip(0, 5).tolist()
    past_dates = df.index.to_list()
    last_month = datetime.strptime(past_dates[-1], "%Y/%m").date()
    future_dates = [add_months(last_month, i) for i in range(1, len(forecasting)+1)]
    
    # past_values: the last year monthly avarage ratings, min: 3 months, max: 12 months
    # past_dates: the year/month associated with each value of the past
    # future_forecasting: the prediction for the next 3 months avarage ratings
    # future_dates: the year/month associated with each value of the forecasted future
    result = {
        "past_values": time_series,
        "past_dates": past_dates,
        "future_forecasting": forecasting,
        "future_dates": future_dates
    }
    

    return result  
