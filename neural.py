import sys
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Used to read all the contents in the directory into a dataframe
# Useful for spark folders
def open_files(folder):
    rows = []
    for path in pathlib.Path(folder).iterdir():
        if path.is_file() and path != pathlib.Path('{}/_SUCCESS'.format(folder)):
            rows.append(pd.read_csv(path))
    merged = pd.concat(rows)
    return merged

# This was just used to generate a random sample of our very large dataset
def random_sample(df):
    df.sample(n=100000).drop(['Unnamed: 0'], axis=1).to_csv('reduced_sample.csv', index=False)
    
    
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

def process_xy(df):
    df['Date' ]= pd.to_datetime(df['Date'], errors='coerce').dt.date
    X = df.drop(['SNo', 'Name', 'Symbol', 'High', 'Low', 'Close', 'Open'], axis=1)
    X = X.drop(['Marketcap'], axis=1)
    y = df.Close
    dates = df.Date
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20)
    
    train_dates = X_train.Date
    valid_dates = X_valid.Date
    
    X_train = X_train.drop('Date', axis=1)
    X_valid = X_valid.drop('Date', axis=1)
    return X_train, X_valid, y_train, y_valid, dates, valid_dates, y

def build_network():
    
    model = Sequential()
    model.add(Dense(units = 20, activation = 'relu', input_dim = 6))
    model.add(Dense(units = 20, activation = 'relu'))
    model.add(Dense(1,))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])
    
    return model

def main(in_dir, in_csv):
    if in_dir is not None:
        df = open_files(in_dir)
        df = process_date(df)
        df2 = pd.read_csv(in_csv)
        df2 = process_currency(df2)
        merged = merge_on_date(df2, df)
        merged = merged.sample(10000)

        X = merged.drop(['High', 'Low', 'Close', 'Open', 'Marketcap'], axis=1)
        y = merged.Close
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20)
        dates = merged.date
        train_dates = X_train.date
        valid_dates = X_valid.date
        X_train = X_train.drop('date', axis=1)
        X_valid = X_valid.drop('date', axis=1)
    else:
        df = pd.read_csv(in_csv)
        smp = df.sample(10000)
        X_train, X_valid, y_train, y_valid, dates, valid_dates, y = process_xy(smp)
        
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)
    
    model = build_network()
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1000, batch_size=10, verbose=0)
        
    y_pred = model.predict(X_valid)
    
    print('valid data score=%.5g' % (mean_squared_error(y_valid, y_pred)))

    plt.figure(figsize=(8, 5))
    plt.scatter(dates, y, label = "Actual Price")
    plt.plot(valid_dates, model.predict(X_valid), 'r.', label = "Predicted Price")
    plt.legend(loc = "upper left")
    plt.savefig('{}/model_nn.png'.format(out_dir))
    #plt.show()
    
if __name__ == '__main__':
    if len(sys.argv) == 4:
        in_dir = sys.argv[1]
        in_csv = sys.argv[2]
        out_dir = sys.argv[3]
        main(in_dir, in_csv)
     
    elif len(sys.argv) == 3:
        in_csv = sys.argv[1]
        out_dir = sys.argv[2]
        main(None, in_csv)
    else:
        print('error, please include data')