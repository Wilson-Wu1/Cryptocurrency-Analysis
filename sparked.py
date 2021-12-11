import sys
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession, DataFrameNaFunctions, functions, types
# from pyspark.sql.functions import pandas_udf, PandasUDFType

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.4' # make sure we have Spark 2.4+

reddit_schema = types.StructType([
    types.StructField('index', types.LongType()),
    types.StructField('datetime', types.StringType()),
    types.StructField('date', types.StringType()),
    types.StructField('author', types.StringType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('created_utc', types.LongType()),
    types.StructField('score', types.FloatType()),
    types.StructField('controversiality', types.IntegerType()),
    types.StructField('body', types.StringType()),
    ])

nltk.download('stopwords')
nltk.download('vader_lexicon')
stopwords = nltk.corpus.stopwords.words("english")

def sent_analysis(a: str):
    sia = SentimentIntensityAnalyzer()
    res = sia.polarity_scores(str(a))
    key = max(res, key=res.get)

    return "{},{},{}".format(res['pos'], res['neg'], res['neu'])

def main(in_directory, out_directory_sum, out_directory_avg, out_directory_all):

    data = spark.read.csv(in_directory,
                          schema=reddit_schema,header=True, sep=',')

    data=data.dropna() # maybe like this?

    counted = data.groupby('subreddit').count()
    counted.cache()
    
    counted = counted.sort(functions.col('count').desc())

    # This will only be 10 rows of 2 columns
    top_reddits = counted.take(10)
    top_list = []

    # Again we're only traversing 10 items so for loops wont be slow
    for row in top_reddits:
        top_list.append(row.asDict()['subreddit'])

    reduced_data = data[data.subreddit.isin(top_list)]
    sent = functions.udf(sent_analysis,
                        returnType=types.StringType())

    sent_reduced = data.select(
        data['date'],
        data['score'],
        data['controversiality'],
        sent('body').alias('sentiments')
        
    )

    final = sent_reduced.withColumn('positive', functions.split(sent_reduced['sentiments'], ',')[0].cast(types.FloatType()))\
        .withColumn('negative', functions.split(sent_reduced['sentiments'], ',')[1].cast(types.FloatType()))\
        .withColumn('neutral', functions.split(sent_reduced['sentiments'], ',')[2].cast(types.FloatType()))

    final = final.drop('sentiments')
    final = final.cache()
    final.write.csv(out_directory_all, mode='overwrite', header=True)
    summed = final.groupby('date').sum()
    avg = final.groupby('date').avg()

    # This is a dangerous operation, but in this case it is safe
    # There is one row per day between the years 2010-2019
    # So there are at most 9*365 rows
    summed.coalesce(1).write.csv(out_directory_sum, mode='overwrite', header=True)
    avg.coalesce(1).write.csv(out_directory_avg, mode='overwrite', header=True)


if __name__=='__main__':
    in_dir = sys.argv[1]
    out_dir_sum = sys.argv[2]
    out_dir_avg = sys.argv[3]
    out_dir_all = sys.argv[4]
    main(in_dir, out_dir_sum, out_dir_avg, out_dir_all)
