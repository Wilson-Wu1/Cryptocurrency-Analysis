import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import datetime 
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor

# This function is for Sentiment analysis 
def sent(text):
    sia = SentimentIntensityAnalyzer()
    res = sia.polarity_scores(str(text))
    key = max(res, key=res.get)
    
    if key == 'neg':
        return -1 * res['neg']
    elif key =='neu':
        return 0
    else:
        return res['pos']

def sentiment_analysis():
    # This code doesn't need to be run, but i will include it so you can see how 
    # I got the 'reddit_results.csv' file

   
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    stopwords = nltk.corpus.stopwords.words("english")

    reddit_data = pd.read_csv('reddit.zip', compression='zip', dtype='object')
    reddit_data = reddit_data.drop(['Unnamed: 0', 'datetime', 'author'], axis=1)

    nan_value = float("NaN")
    reddit_data.replace("", nan_value, inplace=True)
    reddit_data.dropna(inplace=True)

    # this is the code to apply the sentiment analysis
    # It takes a WHILE to run, so I will just include the csv file of the results
    # reddit_data['sentiment'] = reddit_data['body'].apply(sent)
    # reddit_data.to_csv('reddit_results.csv', index=False)

def process_and_merge():
    reddit_data = pd.read_csv('reddit_results.csv', dtype='object')
    group_data = reddit_data.drop(['created_utc', 'body'], axis=1)
    
    # This section turns the string types of the dataset into useable numeric types
    group_data['score'] = pd.to_numeric(group_data['score'], errors='coerce')
    group_data['controversiality'] = pd.to_numeric(group_data['controversiality'], errors='coerce')
    group_data['sentiment'] = pd.to_numeric(group_data['sentiment'], errors='coerce')
    group_data['date']=pd.to_datetime(group_data['date']).dt.date

    # I tried messing around with grouping by two types
    # it might be worth playing around with that
    # grouped = group_data.groupby(['date', 'subreddit']).mean()

    daily_full = group_data.groupby('date').mean()
    
    bit_info = pd.read_csv('coin_Bitcoin.csv', parse_dates=['Date'])
    bit_info = bit_info.drop(['SNo', 'Name', 'Symbol'], axis=1)
    # df['date_column'] = pd.to_datetime(df['datetime_column']).dt.date
    bit_info['Date'] = pd.to_datetime(bit_info['Date']).dt.date
    bit_info = bit_info.groupby('Date').sum()

    merging = daily_full.merge(bit_info, left_index=True, right_index=True)
    
    return merging


def run_model(df):

    X = df.drop(['High', 'Low', 'Close'], axis=1)
    y = df.Close
    
    # Test train split into the first 80% as our train, and last 20% for test
    X_train = X.iloc[0:int(0.8*X.shape[0])]
    X_valid = X.iloc[int(0.8*X.shape[0]):]
    y_train = y.iloc[0:int(0.8*y.shape[0])]
    y_valid = y.iloc[int(0.8*y.shape[0]):]

    estimators = 300
    depth = 7
    split = 50

    model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(n_estimators=estimators,
            max_depth=depth , min_samples_split=split, min_samples_leaf=15)
        )

    model.fit(X_train, y_train)
    # ONLY INCLUDE SCORE IN THE FINAL SUBMISSION
    print(model.score(X_valid, y_valid))
    print(model.score(X_train, y_train))

# uncomment if you want to see the plot
#     plt.figure(figsize=(8, 5))
#     plt.scatter(merging.index, merging.Close)
#     plt.plot(X_valid.index, model.predict(X_valid), 'r-')
#     #plt.plot(merging.Date, deriv, 'g-')
#     plt.show()
    
    
def main():
    
    # sentiment_analysis()
    df = process_and_merge()
    run_model(df)
    
    
if __name__== '__main__':
    main()