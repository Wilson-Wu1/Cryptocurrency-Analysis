import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import pathlib
import glob
from scipy import stats
import os
import itertools


path = str(pathlib.Path().resolve()) + "\coins_data\*.csv"
file_array = []


for path1 in glob.glob(path):
    file_array.append(path1)

        
product = list(itertools.combinations_with_replacement(file_array, 2))

first_coin_array = []
second_coin_array = []
correlation_array = []

for pair in product:

    df1 = pd.read_csv(pair[0])
    df2 = pd.read_csv(pair[1])
    df1['Date'] = pd.to_datetime(df1['Date']).dt.date
    df2['Date'] = pd.to_datetime(df2['Date']).dt.date
    
    #Merge the two dataframes, based on date.
    merged_df = pd.merge(df1, df2, on='Date')
    merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.date

    #Get correlation coefficient
    correlation_coef = stats.linregress(merged_df['Close_x'], merged_df['Close_y']).rvalue
    
        
    print(os.path.split(pair[0])[1], os.path.split(pair[1])[1], correlation_coef)
   
    
