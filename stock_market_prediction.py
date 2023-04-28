# -*- coding: utf-8 -*-
"""Stock Market Prediction (1).ipynb
# Stock Market Prediction Using Numerical & Textual Analysis

## Importing Libraries
"""

import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('vader_lexicon')
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost 
import lightgbm

"""## Load the Textual Data"""

df_text = pd.read_csv("../input/india-headlines-news-dataset/india-news-headlines.csv")
df_text.head()

"""## Handling with Textual Data:"""

df_text.drop(0, inplace=True)
df_text.drop('headline_category', axis = 1, inplace=True)
df_text.head()

df_text["publish_date"] = pd.to_datetime(df_text["publish_date"],format='%Y%m%d')
df_text.info()

df_text.shape

df_text.isnull().sum()

df_text['headline_text'] = df_text.groupby(['publish_date']).transform(lambda x : ' '.join(x)) 
df_text = df_text.drop_duplicates() 
len(df_text)

df_text

df_text.reset_index(inplace=True,drop=True)

"""#### Remove Unwanted Characters from the head_line text

"""

df_text.replace("[^a-zA-Z']"," ",regex=True,inplace=True)
df_text["headline_text"].head(5)

def Subjectivity(text):
  return TextBlob(text).sentiment.subjectivity

def Polarity(text):
  return  TextBlob(text).sentiment.polarity

df_text['Subjectivity'] = df_text['headline_text'].apply(Subjectivity)
df_text['Polarity'] = df_text['headline_text'].apply(Polarity)
df_text

"""### Visualize a Polarity and Subjectivity

"""

plt.figure(figsize = (10,6))
df_text['Polarity'].hist(color = 'green')

plt.figure(figsize = (10,6))
df_text['Subjectivity'].hist(color = 'blue')

"""### Sentiment Analysis using News Headlines



"""

snt = SentimentIntensityAnalyzer()

df_text['Compound'] = [snt.polarity_scores(v)['compound'] for v in df_text['headline_text']]
df_text['Negative'] = [snt.polarity_scores(v)['neg'] for v in df_text['headline_text']]
df_text['Neutral'] = [snt.polarity_scores(v)['neu'] for v in df_text['headline_text']]
df_text['Positive'] = [snt.polarity_scores(v)['pos'] for v in df_text['headline_text']]
df_text

"""## Load the Numerical Data """

df_num = pd.read_csv("../input/numerical-stock-prices/QMCI.csv")
df_num.head()

"""## Handling with Numerical Data

All the things that are going to happen in this data must be related to time to this index.
"""

df_num["Date"] = pd.to_datetime(df_num["Date"],format='%Y-%m-%d')
df_num.info()

"""Let's see how the data will be like:"""

df_num

df_num.describe()

#check for null values
df_num.isnull().sum()

plt.figure(figsize=(10,6))
df_num['Close'].plot()
plt.ylabel('QMCI')

"""### Plotting Moving Average"""

close = df_num['Close']

ma = close.rolling(window = 50).mean()
std = close.rolling(window = 50).std()

plt.figure(figsize=(10,6))
df_num['Close'].plot(color='g',label='Close')
ma.plot(color = 'r',label='Rolling Mean')
std.plot(label = 'Rolling Standard Deviation')

plt.legend()

#split the data to train and test
train = df_num[:200]
test = df_num[200:]

"""### Rolling mean and Standard Deviation

"""

def test_stationarity(timeseries):

 #Determing rolling statistics
 rolmean = timeseries.rolling(20).mean()
 rolstd = timeseries.rolling(20).std()

 #Plot rolling statistics:
 plt.figure(figsize = (10,8))
 plt.plot(timeseries, color = 'y', label = 'original')
 plt.plot(rolmean, color = 'r', label = 'rolling mean')
 plt.plot(rolstd, color = 'b', label = 'rolling std')
 plt.xlabel('Date')
 plt.legend()
 plt.title('Rolling Mean and Standard Deviation',  fontsize = 20)
 plt.show(block = False)
 
 print('Results of dickey fuller test')
 result = adfuller(timeseries, autolag = 'AIC')
 labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
 for value,label in zip(result, labels):
   print(label+' : '+str(value) )
 if result[1] <= 0.05:
   print("Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
 else:
   print("Weak evidence against null hypothesis, time series is non-stationary ")
test_stationarity(train['Close'])

train_log = np.log(train['Close']) 
test_log = np.log(test['Close'])

mav = train_log.rolling(24).mean() 
plt.figure(figsize = (10,6))
plt.plot(train_log) 
plt.plot(mav, color = 'red')

"""## Auto arima to make predictions using log data

"""

!pip install pmdarima

from pmdarima import auto_arima
model = auto_arima(train_log, trace = True, error_action = 'ignore', suppress_warnings = True)
model.fit(train_log)
predictions = model.predict(n_periods = len(test))
predictions = pd.DataFrame(predictions,index = test_log.index,columns=['Prediction'])

"""#### Visualize Prediction"""

plt.plot(train_log, label='Train')
plt.plot(test_log, label='Test')
plt.plot(predictions, label='Prediction')
plt.title('QMCI Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')

rms = np.sqrt(mean_squared_error(test_log,predictions))
print("RMSE : ", rms)

"""## Merging the Numerical and Textual Data

"""

merge = df_text
merge

data = merge[['Subjectivity', 'Polarity', 'Compound', 'Negative', 'Neutral' ,'Positive']]
data

X = data[:252]
y = df_num['Close']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
x_train.shape

"""## Apply Models

### RandomForestRegressor Model
"""

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
prediction=rf.predict(x_test)

print(prediction[:10])
print(y_test[:10])
print('Mean Squared error: ',mean_squared_error(prediction,y_test))

"""### DecisionTreeRegressor Model"""

from sklearn.tree import DecisionTreeRegressor
dec_tree = DecisionTreeRegressor()
dec_tree.fit(x_train, y_train)
predictions = dec_tree.predict(x_test)
print('Mean Squared error: ',mean_squared_error(predictions,y_test))

"""### XGBRegressor Model"""

xgb = xgboost.XGBRegressor()
xgb.fit(x_train, y_train)
predictions = xgb.predict(x_test)
print('Mean Squared error: ',mean_squared_error(predictions,y_test))

"""## Conclusion:
we can see that RandomForestRegressor shows a better performance than the others
"""