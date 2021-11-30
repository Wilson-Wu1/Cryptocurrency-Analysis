import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import datetime 
import sys
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ---  THESE FUNCTIONS ARE JUST FOR REFERENCE ---- #

# This function is for Sentiment analysis 
# NOTE: This code is now handled by the sparked.py
# temporarily keeping for reference 
# def sent(text):
#     sia = SentimentIntensityAnalyzer()
#     res = sia.polarity_scores(str(text))
#     key = max(res, key=res.get)
    
#     if key == 'neg':
#         return -1 * res['neg']
#     elif key =='neu':
#         return 0
#     else:
#         return res['pos']

# NOTE: This code is now handled by the sparked.py 
# temporarily keeping for reference
# def sentiment_analysis():
#     # This code doesn't need to be run, but i will include it so you can see how 
#     # I got the 'reddit_results.csv' file

   
#     nltk.download('stopwords')
#     nltk.download('vader_lexicon')
#     stopwords = nltk.corpus.stopwords.words("english")

#     reddit_data = pd.read_csv('reddit.zip', compression='zip', dtype='object')
#     reddit_data = reddit_data.drop(['Unnamed: 0', 'datetime', 'author'], axis=1)

#     nan_value = float("NaN")
#     reddit_data.replace("", nan_value, inplace=True)
#     reddit_data.dropna(inplace=True)

#     # this is the code to apply the sentiment analysis
#     # It takes a WHILE to run, so I will just include the csv file of the results
#     # reddit_data['sentiment'] = reddit_data['body'].apply(sent)
#     # reddit_data.to_csv('reddit_results.csv', index=False)

# # Most of the processing is now already taken care of by the sparked.py
# # temporarily keeping for reference
# def process_and_merge():
#     reddit_data = pd.read_csv('reddit_results.csv', dtype='object')
#     group_data = reddit_data.drop(['created_utc', 'body'], axis=1)
    
#     # This section turns the string types of the dataset into useable numeric types
#     group_data['score'] = pd.to_numeric(group_data['score'], errors='coerce')
#     group_data['controversiality'] = pd.to_numeric(group_data['controversiality'], errors='coerce')
#     group_data['sentiment'] = pd.to_numeric(group_data['sentiment'], errors='coerce')
#     group_data['date']=pd.to_datetime(group_data['date']).dt.date

#     # I tried messing around with grouping by two types
#     # it might be worth playing around with that
#     # grouped = group_data.groupby(['date', 'subreddit']).mean()

#     daily_full = group_data.groupby('date').mean()
    
#     bit_info = pd.read_csv('coin_Bitcoin.csv', parse_dates=['Date'])
#     bit_info = bit_info.drop(['SNo', 'Name', 'Symbol'], axis=1)
#     # df['date_column'] = pd.to_datetime(df['datetime_column']).dt.date
#     bit_info['Date'] = pd.to_datetime(bit_info['Date']).dt.date
#     bit_info = bit_info.groupby('Date').sum()

#     merging = daily_full.merge(bit_info, left_index=True, right_index=True)
    
#     return merging

# ------------------------------------------------ #


# Input: Takes in dataframe of 'coin_Bitcoin.csv' or other related coin datasets
# Output: A dataframe with added labels for increasing/not increasing
def create_labels(df):
    df['NextClose'] = df['Close'].shift(-1)
    df.loc[df['NextClose'] > df['Close'], 'Change'] = 'Increase'
    df.loc[df['NextClose'] <= df['Close'], 'Change'] = 'No Increase'
    df = df.dropna()
    df = df.drop('NextClose', axis=1)
    
    vals = df.groupby('Change').count()
    ratio = vals.loc['Increase']['SNo'] / (vals.loc['Increase']['SNo'] + vals.loc['No Increase']['SNo'])
    
    # Show the ratio of Increase to No Increase rows. This will indicate if the dataset is inbalanced 
    # if it far from a 50/50 split
    print(ratio, 1-ratio)
    
    return df

# Input: Takes in dataframe with column 'date
# Output: Turns the date column into datetime object 
def process_date(df):
    df['date' ]= pd.to_datetime(df['date'], errors='coerce').dt.date
    return df

# Input: Dataframe from the currency (eg. coin_Bitcoin.csv) datasets
# Output: removes unused columns and sets index to date
def process_currency(df):
    bit_info = df.drop(['SNo', 'Name', 'Symbol'], axis=1)
    bit_info = bit_info.rename(columns={'Date': 'date'})
    bit_info['date'] = pd.to_datetime(bit_info['date']).dt.date
    return bit_info
    

# Input: Two dataframes with 'date' column
# Output: The result of merging the two dataframes
def merge_on_date(df1, df2):
    merging = df1.merge(df2, on='date')
    return merging

# Input: A dataframe post merging, an output DIRECTORY
# Output: runs random forest regression to predict closing price
def random_forest_regression(df, out_dir):
    
    X = df.drop(['High', 'Low', 'Close', 'Open', 'Volume', 'Marketcap', 'Change'], axis=1)
    y = df.Close
    dates = df.date
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
    
    train_dates = X_train.date
    valid_dates = X_valid.date
    
    X_train = X_train.drop('date', axis=1)
    X_valid = X_valid.drop('date', axis=1)
    
    # Test train split into the first 80% as our train, and last 20% for test
    #X_train = X.iloc[0:int(0.8*X.shape[0])]
    #X_valid = X.iloc[int(0.8*X.shape[0]):]
    #y_train = y.iloc[0:int(0.8*y.shape[0])]
    #y_valid = y.iloc[int(0.8*y.shape[0]):]

    estimators = 300
    depth = 7
    split = 50

    model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(n_estimators=estimators,
            max_depth=depth , min_samples_split=split, min_samples_leaf=15)
        )

    model.fit(X_train, y_train)

    print(model.score(X_valid, y_valid))
    print(model.score(X_train, y_train))

    plt.figure(figsize=(8, 5))
    plt.scatter(dates, y)
    plt.plot(valid_dates, model.predict(X_valid), 'r.')
    plt.savefig('{}/model.png'.format(out_dir))

    
    
def main():
    
    # To run:
    # create output directory for .png images
    
    # python3 Model.py coin_Bitcoin.csv DailyAverages.csv output
    # OR
    # python3 Model.py coin_Bitcoin.csv DailySum.csv output
    
    # for now we are just merging two datasets
    in_dir1 = sys.argv[1] # coin_Bitcoin.csv (or other currency datasets)
    in_dir2 = sys.argv[2] # one of (DailyAverages.csv, DailySum.csv)
    out_dir = sys.argv[3] # folder to save images
    
    df_bitcoin = pd.read_csv(in_dir1, parse_dates=['Date'])
    df_reddit = pd.read_csv(in_dir2, parse_dates=['date'])
    
    # Pre processing for coin_Bitcoin.csv (or other currency datasets)
    df_bitcoin = create_labels(df_bitcoin)
    df_bitcoin = process_currency(df_bitcoin)
    df_bitcoin = process_date(df_bitcoin)
    
    # Pre processing for one of (DailyAverages.csv, DailySum.csv)
    df_reddit = process_date(df_reddit)
    
    # Merges the two datasets on 'date'
    merged = merge_on_date(df_bitcoin, df_reddit)
    random_forest_regression(merged, out_dir)
    
    
if __name__== '__main__':
    main()