import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import glob
from scipy import stats
import os
import seaborn as sns

##########################################################################
# Creates a heatmap containing correlation coef. for all pairs of coins. #
##########################################################################


path = str(pathlib.Path().resolve()) + "\coins_data\*.csv"
file_array = []
for path1 in glob.glob(path):
    file_array.append(path1)


#Arrays for creating dataframe
column_names = []
values = []

#Loop through every pair of coin and get correlation coef.
for i in file_array:
    #Parse the file name to just get coin/token name
    coin_name = os.path.split(i)[1]
    coin_name = coin_name.split("_")[1]
    coin_name = coin_name.split(".")[0]
    coin_name = coin_name.split("Coin")[0]
    column_names.append(coin_name)

    
    for j in file_array:
        df1 = pd.read_csv(i)
        df2 = pd.read_csv(j)
        df1['Date'] = pd.to_datetime(df1['Date']).dt.date
        df2['Date'] = pd.to_datetime(df2['Date']).dt.date
        
        #Merge the two dataframes, based on date.
        merged_df = pd.merge(df1, df2, on='Date')
        merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.date
        
        #Get correlation coefficient, and move into values array
        correlation_coef = stats.linregress(merged_df['Close_x'], merged_df['Close_y']).rvalue
        values.append(correlation_coef)

        #print(os.path.split(i)[1], os.path.split(j)[1], correlation_coef)

#Move correlation values into numpy array and reshape for dataframe/heatmap
values = np.array(values)
values = values.reshape(23,23)

#Create dataframe
df = pd.DataFrame(data = values, columns = column_names, index = column_names)
df = df.drop(columns = ['Cosmos', 'Tron','Dogecoin','EOS','Uniswap','Litecoin','Stellar','Cryptocom','Monero',"Iota","NEM","Aave"])

#Plotting...
plt.rcParams["figure.figsize"] = (20,10)
plt.title("Correlation Coefficient", fontsize=20)
sns.heatmap(df, annot = True)
plt.xticks(rotation=45, fontsize = 15)
plt.yticks(fontsize = 10)
plt.savefig('correlation_coef.jpg')
#plt.show()


##########################################################################
#Get the correlations against bitcoin, and sort

bitcoin_correlation = df["Bitcoin"] 
bitcoin_correlation = bitcoin_correlation.sort_values(ascending = False)
bitcoin_correlation = bitcoin_correlation.drop(labels = ["USDC","Bitcoin"])


#Plotting...
plt.clf()
plt.rcParams["figure.figsize"] = (20,20)
plt.title("Correlation Coefficient against Bitcoin", fontsize=20)
plt.ylabel("Correlation Coefficient", fontsize=15)
plt.xlabel("CryptoCurrency", fontsize=15)
bitcoin_correlation = bitcoin_correlation.round(decimals = 2)
plt.bar(bitcoin_correlation.index, bitcoin_correlation, color = (0.090, 0.188, 0.690, 0.75))

#Ref: https://stackoverflow.com/questions/40287847/python-matplotlib-bar-chart-adding-bar-titles
ax = plt.gca()
plt.bar_label(ax.containers[0])
plt.xticks(rotation=45, fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid()

#For some reason savefig cuts off the x-label. plt.show() is used instead.
#plt.savefig('bitcoin_coef.jpg')
plt.show()



##########################################################################
#Get the sums of the correlation coef 
for i in df:
    df[i] = df[i].abs()

df["sum"] = df.sum(axis=1)
#print(df)
