from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkContext, SparkConf

import pandas as pd
import calendar
from pyspark.sql.functions import col, when, year, month, dayofmonth, hour, date_format
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
    
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.load('/user/edureka_854312/certificate_project/train.csv',
                     format='csv', inferSchema=True, header=True)
    
    df = df.withColumn('label', df['count']) 
    
    # create weekday
    df_weekday = df.select('datetime', date_format('datetime', 'u').\
                        alias('weekday')).withColumn('datetime2', df.datetime).\
                        drop('datetime')
    df = df.join(df_weekday, df.datetime == df_weekday.datetime2).drop('datetime2')    
    
    # Extract features from datetime
    df = df.withColumn('year', year(df.datetime))
    df = df.withColumn('month', month(df.datetime))
    df = df.withColumn('day', dayofmonth(df.datetime))
    df = df.withColumn('hour', hour(df.datetime))
    
    #df = df.fillna(1)
    trainingData, testData = df.randomSplit([0.7, 0.3], seed=123)
    
    # create categorical variables
        # Note: In streaming, using OneHotEncoder without StringIndexer will cause error. 
            # But it's fine to use StringIndexer without OneHotEncoder if the variable is binary.
    seasonIndexer = StringIndexer(inputCol='season', outputCol='seasonIndex')
    seasonEncoder = OneHotEncoder(inputCol='seasonIndex', outputCol='seasonVec')
    holidayIndexer = StringIndexer(inputCol='holiday', outputCol='holidayIndex')
#     holidayEncoder = OneHotEncoder(inputCol='holiday', outputCol='holidayVec')
    workingdayIndexer = StringIndexer(inputCol='workingday', outputCol='workingdayIndex')
#     workingdayEncoder = OneHotEncoder(inputCol='workingday', outputCol='workingdayVec')
    weatherIndexer = StringIndexer(inputCol='weather', outputCol='weatherIndex')
    weatherEncoder = OneHotEncoder(inputCol='weatherIndex', outputCol='weatherVec')
    weekdayIndexer = StringIndexer(inputCol='weekday', outputCol='weekdayIndex')
    weekdayEncoder = OneHotEncoder(inputCol='weekdayIndex', outputCol='weekdayVec')
    
    # Create feature
    vectorAssembler = VectorAssembler(inputCols = ['temp', 'atemp', 'humidity', 'windspeed',
                                                   'holidayIndex', 'workingdayIndex', 
                                                   'seasonVec', 
                                                   'weatherVec', 
                                                   'year', 'month', 'day', 'weekdayVec', 
                                                   'hour'],
                                      outputCol = 'features')
    
    gbt = GBTRegressor(featuresCol='features', labelCol = 'label', maxIter=3)
    
    # Build pipeline
    pipeline = Pipeline(stages=[seasonIndexer, seasonEncoder, 
                                holidayIndexer, workingdayIndexer, 
                                weatherIndexer, weatherEncoder, 
                                weekdayIndexer, weekdayEncoder,
                                vectorAssembler, gbt
                               ])
    model = pipeline.fit(trainingData)
    
    # Make prediction
    predictions = model.transform(testData)
    predictions.select("prediction", "count", "features").show(10)

    evaluator = RegressionEvaluator(labelCol="count", predictionCol="prediction", metricName="r2")
    rmse = evaluator.evaluate(predictions)
    print("=== R2 on test data = %g ===" % rmse)

    evaluator = RegressionEvaluator(labelCol="count", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("=== RMSE on test data = %g ===" % rmse)

    model.save('/user/edureka_854312/certificate_project/model/')