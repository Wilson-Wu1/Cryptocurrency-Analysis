import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import pathlib
from scipy import stats



#Compare the percentage change between two coins
def compare_correlation(first_coin, second_coin):


    path1 = str(pathlib.Path().resolve()) + "\coins_data\coin_" + first_coin + ".csv"
    path2 = str(pathlib.Path().resolve()) + "\coins_data\coin_" + second_coin + ".csv"
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    #Convert Date column to datetime
    df1['Date'] = pd.to_datetime(df1['Date']).dt.date
    df2['Date'] = pd.to_datetime(df2['Date']).dt.date

    #Calculate percentage changes
    df1['Price'] = 100 * ( (df1['Close'] - df1['Open']) / df1['Open'].abs() )
    df2['Price'] = 100 * ( (df2['Close'] - df2['Open']) / df2['Open'].abs() )

    #Merge the two dataframes, based on date.
    merged_df = pd.merge(df1, df2, on='Date')
    merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.date
    
    #Get correlation coefficient
    correlation_coef = stats.linregress(merged_df['Price_x'], merged_df['Price_y']).rvalue
    
    #Plotting...
    plt.rcParams["figure.figsize"] = (20,10)
    plt.plot(merged_df['Date'], merged_df['Price_x'], color = 'black')
    plt.plot(merged_df['Date'], merged_df['Price_y'],  alpha=0.75 )
    plt.title("Percentage Change of " + first_coin + " vs. " + second_coin + "    Coorelation Coefficient: " +  str(correlation_coef))
    plt.xlabel("Date")
    plt.ylabel("% Change")
    plt.legend(labels = [first_coin, second_coin])
    plt.grid()

    plt.show()




first_coin = sys.argv[1]
second_coin = sys.argv[2]
compare_correlation(first_coin, second_coin)