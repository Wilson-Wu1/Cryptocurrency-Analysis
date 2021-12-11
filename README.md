#  CMPT 353 Final Project - Cryptocurrency Analysis

## How to Run Programs

### run reddit.py

Required Libraries: Pandas, Numpy, psaw, tqdm

This is the API interface we used to collect the reddit data. This can be run with:

  python3 reddit.py out_dir
  
When run this will collect all reddit comments that contain the word "bitcoin" between 2009 and 2012. 
This can be changed to find data about other coins or over a larger time span.

The directory produced by this code will not be used in later parts of this assignment as we will include relevant csv files


### run spark.py

Required Libraries: nltk

This program is what we used to run on our coin_Bitcoin.csv file which is quite large (~2.3Gb) we included a smaller 
reduced_sample.csv to run. 

This program is what runs the sentiment analysis. It will apply the sentiment analysis to the body of the comments. 
Afterwards the program will output directories for the data by average daily sentiment, and a directory of unaggregated data.

To run spark-submit spark.py reduced_sample.csv out_directory_average out_directory_all

The csv file in out_directory_average will be used for the regression training models, and the classification training models

The out_directory_all run with the Neural Net 

### run regression models

### run classification models

### Neural Net

Required Libraries: Pandas, Numpy, sys, matplotlib, pathlib, sklearn, keras

This program will run in two different ways depending on how you call it

if you run:

python3 neural.py out_directory_all coin_Bitcoin.csv

then this will run the neural net on the provided sample data

python3 neural.py NeuralNetInput.csv

then this will run the neural net on the full dataset

NeuralNetInput.csv can be found at the provided google drive

### Run correlation.py
Required Libraries: Pandas, Numpy, sys, matplotlib, pathlib, scipy.stats

This program requires the coins .csv files to be located in the same directory level as the program.

Run the program by executing "python correlation.py" in the terminal.
Both the heatmap and bitcoin correlation visuals will be saved as jpgs.

Please note that the bitcoin correlation graph will also show when running the program. This is because the x-label is cut-off while saving it as a figure.

### Run precent_change.py
Required Libraries: Pandas, sys, matplotlib, pathlib, scipy.stats

This program requires the coins .csv files to be located in the same directory level as the program.

Run the program by executing "python percent_change.py [firstCoin] [secondCoin]" in the terminal. Ensure to indicate the two coins that are desired as sys arguments. The visuals will be saved as .jpgs.

Example: >python percent_change.py bitcoin polkadot
