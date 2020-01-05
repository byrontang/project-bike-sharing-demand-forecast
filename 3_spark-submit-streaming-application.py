from __future__ import print_function

from pyspark.sql import SparkSession, SQLContext, Row
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

import pandas as pd
import calendar
import datetime
from pyspark.sql.functions import col, when, year, month, dayofmonth, hour, date_format, unix_timestamp
from pyspark.sql.types import TimestampType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel

if __name__ == "__main__":
    
    # Spark Streaming
    sc = SparkContext(appName="bikeSharingDemand")
    ssc = StreamingContext(sc, 120)
    
    now = datetime.datetime.now()
    filepath = "/user/edureka_854312/certificate_project/" + now.strftime("%Y-%m-%d/")
    print("filepath:", filepath)
    lines = ssc.textFileStream(filepath)
    
    def process(t, rdd):
        if rdd.isEmpty():
            print("filepath:", filepath)
            print("==== EMPTY ====")
            return

        print("=== RDD Found ===")
        spark = SparkSession.builder.getOrCreate()
        parts = rdd.map(lambda l: l.split(','))
        rowRdd = parts.map(lambda x: Row(datetime_str=x[0], season=int(x[1]), holiday=int(x[2]), 
                                         workingday=int(x[3]), weather=int(x[4]), temp=float(x[5]), 
                                         atemp=float(x[6]), humidity=float(x[7]), windspeed=float(x[8])))           
        df = spark.createDataFrame(rowRdd)
        print("=== df ===")
        print(df.show())
        
        # Convert datetime
        df_datetime = df.select('datetime_str', 
                                unix_timestamp('datetime_str', "MM/dd/yyyy HH:mm").cast(TimestampType()).alias("datetime"))
        print(df_datetime.show())
        df = df.join(df_datetime, df.datetime_str == df_datetime.datetime_str).drop('datetime_str')
        
        # Create weekday
        df = df.withColumn('datetime', df_datetime.datetime)
        df_weekday = df.select('datetime', date_format('datetime', 'u').\
                        alias('weekday')).withColumn('datetime2', df.datetime).\
                        drop('datetime')
        print(df_weekday.show())
        df = df.join(df_weekday, df.datetime == df_weekday.datetime2).drop('datetime2')
        
        # Extract features from datetime
        df = df.withColumn('year', year(df.datetime))
        df = df.withColumn('month', month(df.datetime))
        df = df.withColumn('day', dayofmonth(df.datetime))
        df = df.withColumn('hour', hour(df.datetime))
        
        print("=== Data Frame ===")
        print(df.show())
        
        if not rdd.isEmpty():
            #df = df.withColumn('type', col('Type'))
            # load saved model
            model = PipelineModel.load('/user/edureka_854312/certificate_project/model/')
            # Predict count
            predictions = model.transform(df)
            print("=== Prediction ===")
            print(predictions.show())
    
    lines.pprint()  
    lines.foreachRDD(process)
    
    ssc.start()
    ssc.awaitTermination()
