# project-bike-sharing-demand-forecast

The project uses the [Bike Sharing Demend Dataset on Kaggle](https://www.kaggle.com/c/bike-sharing-demand/data). Regression models in this project are built on PySpark platform. Eventually, the prediction of test dataset is submitted to Kaggle for evaluation. With PySpark, additional works are also put in to develop an application that could run the model on HDFS, stream new data through flume, and make predictions in real-time.

There's still room to develop my best model, and more works for prediction improvements will be done in the future. Below is an overview of the current progress.

## Overview
### I. Data Transformation and Feature Extraction
- Add Weekday Column
- Convert Categorical Variables
- Check Missing Values
- Extract Date and Time Features

### II. Model Development
- Linear Regression
- Random Forest
- Gradient-Boosted Tree Regression
- Best Model

### III. Predict New Data
