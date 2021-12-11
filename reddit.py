from psaw import PushshiftAPI
import datetime as dt
import pandas as pd
import json
import sys
from tqdm import tqdm

api = PushshiftAPI()

# code to interact with PSAW found at
# https://melaniewalsh.github.io/Intro-Cultural-Analytics/04-Data-Collection/14-Reddit-Data.html

# This will just be used as an example of how to get data from Reddit
# we will just take a subset of the data for efficiency sake.

start_epoch = int(dt.datetime(2009, 1, 1).timestamp())
end_epoch = int(dt.datetime(2012, 1, 1).timestamp())

# How to run
# python3 reddit.py OUTPUT_CSV_NAME

def get_data(out_dir):
    
    api_request_generator = api.search_comments(q='bitcoin', after=start_epoch, before=end_epoch)
    
    # Generate the dataframe from the api request
    bitData = pd.DataFrame([submission.d_ for submission in tqdm(api_request_generator)])
    bitData['date'] = pd.to_datetime(bitData['created_utc'], utc=True, unit='s')
    
    # take a subset of the data for the information we need
    returnData = bitData[['author','subreddit', 'date', 'score', 'controversiality', 'body']]
    returnData['date'] = pd.to_datetime(returnData['date']).dt.date
    returnData.to_csv(out_dir, index=True)
    
def main():
    out_dir = sys.argv[1]
    get_data(out_dir)
    
    
    
if __name__ == '__main__':
    main()