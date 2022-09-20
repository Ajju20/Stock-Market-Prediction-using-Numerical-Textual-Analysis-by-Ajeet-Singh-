#!/usr/bin/env python
# coding: utf-8

# # Ajeet Singh Spark Foundation Data Science Intern 

# # Stock Market Prediction using Numerical and Textual Analysis
# 

# **Stock to analyze and predict SENSEX (S&P BSE SENSEX)**
# 
# # Objective: 
# 
# **Create a hybrid model for stock price/performance prediction
# using numerical analysis of historical stock prices and sentimental analysis of
# news headlines.** 

# In[1]:


pip install yfinance --upgrade --no-cache-dir


# In[2]:


#import libraries 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import altair as alt  #Altair is a declarative statistical visualization library for Python

import statsmodels.api as sm 

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import RandomForestRegressor


# In[3]:


#ignoring the warnings
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# import yfinance as yf
# bse_data = yf.download('^BSESN', start='2015-01-01', end='2020-11-03')
# unseenbse_data = yf.download('^BSESN', start='2020-11-03', end='2020-11-04')


# In[5]:


import yfinance as yf
bse_data = yf.download('^BSESN', start='2015-01-01', end='2020-06-30')
#since our Textual Analysis dataset containing news from Times of India News Headlines is only till 30th June 2020. 
#So we will assume today is 29th June 2020 and tomorrow is 30th June 2020. And we have to predict the stock price ((high+low+close)/3) and closing price of BSE index 
#for tomorrow 30th June 2020.
unseenbse_data = yf.download('^BSESN', start='2020-06-30', end='2020-07-01')


# In[6]:


bse_data.columns


# In[7]:


unseenbse_data.columns


# In[8]:


bse_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adjclose', 'Volume': 'volume'}, inplace = True)


# In[9]:


unseenbse_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adjclose', 'Volume': 'volume'}, inplace = True)


# In[10]:


bse_data.info()


# In[11]:


unseenbse_data.info()


# In[12]:


bse_data.head()


# In[13]:


bse_data.tail()


# In[14]:


unseenbse_data.head()


# In[15]:


bse_data.reset_index(inplace=True)


# In[16]:


bse_data.rename(columns={'Date': 'date'}, inplace = True)


# In[17]:


bse_data.head()


# In[18]:


unseenbse_data.reset_index(inplace=True)


# In[19]:


unseenbse_data.rename(columns={'Date': 'date'}, inplace = True)


# In[20]:


unseenbse_data.head()


# In[21]:


bse_data['date'] = pd.to_datetime(bse_data['date'], format = '%Y%m%d')


# In[22]:


unseenbse_data['date'] = pd.to_datetime(unseenbse_data['date'], format = '%Y%m%d')


# In[23]:


#before moving forward let us calculate first the actual price
unseenbsedata_price = round((unseenbse_data['high'] + unseenbse_data['low'] + unseenbse_data['close'])/ 3, 2)
unseenbsedata_price  #actual price


# # Rolling window analysis of time series
# 
# Creating 4,16, 52 week moving average of closing price of BSE index
# 

# In[24]:


def stock_weekmovingavg(wks, df):
  dateclose_data = pd.DataFrame({'date': df['date'], 'close':df['close']})
  dateclose_data.set_index('date', inplace=True)
  num = wks * 5                                 #calculating the number of days in the week. 5 days because BSE is open for 5 days / week
  dateclose_data['movingavg'] = dateclose_data['close'].rolling(window=num).mean().shift()
  return dateclose_data.dropna()


# In[25]:


stock_weekmovingavg(4, bse_data).head()


# In[26]:


stock_weekmovingavg(4, bse_data).plot()


# In[27]:


altdata_fourweek = stock_weekmovingavg(4, bse_data)
altdata_fourweek.reset_index(inplace=True)
altdata_fourweek.rename(columns={list(altdata_fourweek)[0]:'date'}, inplace=True)


# In[28]:


alt.Chart(altdata_fourweek).mark_point().encode(
    x='date',
    y='movingavg'
)


# In[29]:


plotfourweek = altdata_fourweek.filter(['date', 'movingavg'], axis=1) #df.copy()
plotfourweek.index = pd.Index(sm.tsa.datetools.dates_from_range('2015', length=len(altdata_fourweek['date']))) 
del plotfourweek['date']
sm.graphics.tsa.plot_pacf(plotfourweek.values.squeeze())
plt.show()


# In the partial autocorrelation plot above, we have statistically significant partial autocorrelations at lag values 4 and 32. Since it is less than 0 and more than -1 so 4 and 32 represents a perfect negative correlation. While the rest of values are very close to 0 and under the confidence intervals, which are represented as blue shaded regions (which is not vividly seen in the above plot)

# In[30]:


stock_weekmovingavg(16, bse_data).head()


# In[31]:


stock_weekmovingavg(16, bse_data).plot()
plt.show()


# In[32]:


altdata_sixteenweek = stock_weekmovingavg(16, bse_data)
altdata_sixteenweek.reset_index(inplace=True)
altdata_sixteenweek.rename(columns={list(altdata_sixteenweek)[0]:'date'}, inplace=True)


# In[33]:


alt.Chart(altdata_sixteenweek).mark_point().encode(
    x='date',
    y='movingavg'
)


# In[34]:


plotsixteenweek = altdata_sixteenweek.filter(['date', 'movingavg'], axis=1) #df.copy()
plotsixteenweek.index = pd.Index(sm.tsa.datetools.dates_from_range('2015', length=len(altdata_sixteenweek['date']))) 
del plotsixteenweek['date']
sm.graphics.tsa.plot_pacf(plotsixteenweek.values.squeeze())
plt.show()


# In the partial autocorrelation plot above, we have statistically significant partial autocorrelations at lag values 0, 1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28.
# Where 0, 1, 19 represents a perfect positive correlation and 20 represents a perfect negative correlation.
# While the rest of values are very close to 0 and under the confidence intervals, which are represented as blue shaded regions (which is not vividly seen in the above plot)

# In[35]:


stock_weekmovingavg(52, bse_data).head()


# In[36]:


stock_weekmovingavg(52, bse_data).plot()
plt.show()


# In[37]:


altdata_fiftytwoweek = stock_weekmovingavg(52, bse_data)
altdata_fiftytwoweek.reset_index(inplace=True)
altdata_fiftytwoweek.rename(columns={list(altdata_fiftytwoweek)[0]:'date'}, inplace=True)


# In[38]:


alt.Chart(altdata_fiftytwoweek).mark_point().encode(
    x='date',
    y='movingavg'
)


# In[39]:


plotfiftytwoweek = altdata_fiftytwoweek.filter(['date', 'movingavg'], axis=1) #df.copy()
plotfiftytwoweek.index = pd.Index(sm.tsa.datetools.dates_from_range('2015', length=len(altdata_fiftytwoweek['date']))) 
del plotfiftytwoweek['date']
sm.graphics.tsa.plot_pacf(plotfiftytwoweek.values.squeeze())
plt.show()


# In the partial autocorrelation plot above, we have statistically significant partial autocorrelations at lag values 0, 1 representing a perfect positive correlation. While the rest of values are very close to 0 and under the confidence intervals, which are represented as blue shaded regions 

# ## Creating a rolling window of size 10 and 50 of the BSE index

# In[40]:


def rollingwindows(days, df):
  data = df.filter(['date','open','high','low','close'], axis=1) #df.copy()
  data.set_index('date', inplace=True)
  rollingwindows_data = data.rolling(window=days).mean().shift()
  return rollingwindows_data.dropna()


# In[41]:


rollingwindows(10, bse_data).head()


# In[42]:


rollingwindows(10, bse_data).plot()


# In[43]:


altdata_tendays = rollingwindows(10, bse_data)
altdata_tendays.reset_index(inplace=True)
altdata_tendays.rename(columns={list(altdata_tendays)[0]:'date'}, inplace=True)


# In[44]:


alt.Chart(altdata_tendays).mark_point().encode(
    x ='date',
    y = 'close'
)


# In[45]:


rollingwindows(50, bse_data).head()


# In[46]:


rollingwindows(50, bse_data).plot()
plt.show()


# In[47]:


altdata_fiftydays = rollingwindows(50, bse_data)
altdata_fiftydays.reset_index(inplace=True)
altdata_fiftydays.rename(columns={list(altdata_fiftydays)[0]:'date'}, inplace=True)


# In[48]:


alt.Chart(altdata_fiftydays).mark_point().encode(
    x='date',
    y='close'
)


# # Creating the dummy time series:
# 
# Volume shocks : we will be creating a 0/1 dummy-coded boolean time series for shock, based on whether volume traded is 10% higher/lower than previous day. ( 0/1 dummy-coding is for direction of shock)

# In[49]:


def boolean_shock(percent, df, col):
  data = df.filter(['date', col], axis=1) #df.copy()
  data.set_index('date', inplace=True)
  data['percentchg'] = (data[col].pct_change()) * 100  #percentage change compare to previous volume using pct_change() function
  data['shock'] = data['percentchg'].apply(lambda x: 1 if x >= percent else 0)
  data.drop(col, axis = 1, inplace = True)
  return data.dropna()


# In[50]:


boolean_shock(10, bse_data, 'volume')


# In[51]:


altdata_volpercentchg = boolean_shock(10, bse_data, 'volume')
altdata_volpercentchg.reset_index(inplace=True)
altdata_volpercentchg.rename(columns={list(altdata_volpercentchg)[0]:'date'}, inplace=True)


# In[52]:


alt.Chart(altdata_volpercentchg).mark_point().encode(
    x='date',
    y='percentchg'
)


# In[53]:


plotvolpercentchg = altdata_volpercentchg.filter(['date', 'percentchg'], axis=1) #df.copy()
plotvolpercentchg.index = pd.Index(sm.tsa.datetools.dates_from_range('2015', length=len(altdata_volpercentchg['date']))) 
del plotvolpercentchg['date']
sm.graphics.tsa.plot_pacf(plotvolpercentchg.values.squeeze())
plt.show()


# In the partial autocorrelation plot above, we have statistically significant partial autocorrelations at lag values 0, 3, 4, 5, 8, 9. 10, 12, 13, 15, 16, 18, 19, 20, 22, 23, 29, 30, 32. And lag value 0 represents a perfect positive correlation. While the rest of values are very close to 0 and under the confidence intervals, which are represented as blue shaded regions

# In[54]:


boolean_shock(2, bse_data, 'close')


# In[55]:


altdata_closepercentchg2 = boolean_shock(2, bse_data, 'close')
altdata_closepercentchg2.reset_index(inplace=True)
altdata_closepercentchg2.rename(columns={list(altdata_closepercentchg2)[0]:'date'}, inplace=True)


# In[56]:


alt.Chart(altdata_closepercentchg2).mark_point().encode(
    x='date',
    y='percentchg'
)


# In[57]:


plotclosepercentchg2 = altdata_closepercentchg2.filter(['date', 'percentchg'], axis=1) #df.copy()
plotclosepercentchg2.index = pd.Index(sm.tsa.datetools.dates_from_range('2015', length=len(altdata_closepercentchg2['date']))) 
del plotclosepercentchg2['date']
sm.graphics.tsa.plot_pacf(plotclosepercentchg2.values.squeeze())
plt.show()


# In the partial autocorrelation plot above, we have statistically significant partial autocorrelations at lag values 0, 5, 6, 7, 10, 11, 24. And lag value 0 represents a perfect positive correlation. While the rest of values are very close to 0 and under the confidence intervals, which are represented as blue shaded regions

# In[58]:


boolean_shock(10, bse_data, 'close')


# In[59]:


altdata_closepercentchg10 = boolean_shock(10, bse_data, 'close')
altdata_closepercentchg10.reset_index(inplace=True)
altdata_closepercentchg10.rename(columns={list(altdata_closepercentchg10)[0]:'date'}, inplace=True)


# In[60]:


alt.Chart(altdata_closepercentchg10).mark_point().encode(
    x='date',
    y='percentchg'
)


# In[61]:


plotclosepercentchg10 = altdata_closepercentchg10.filter(['date', 'percentchg'], axis=1) #df.copy()
plotclosepercentchg10.index = pd.Index(sm.tsa.datetools.dates_from_range('2015', length=len(altdata_closepercentchg10['date']))) 
del plotclosepercentchg10['date']
sm.graphics.tsa.plot_pacf(plotclosepercentchg10.values.squeeze())
plt.show()


# In the partial autocorrelation plot above, we have statistically significant partial autocorrelations at lag values 0, 5, 6, 7, 10, 11, 24. And lag value 0 represents a perfect positive correlation. While the rest of values are very close to 0 and under the confidence intervals, which are represented as blue shaded regions

# ## Pricing shock without volume shock

# In[62]:


def priceboolean_shock(percent, df):
  df['date'] = pd.to_datetime(df['date'])
  data = df.filter(['date', 'high', 'low','close'], axis=1) #df.copy()
  data.set_index('date', inplace=True)
  data['priceavg'] = (data['high'] + data['low'] + data['close']) / 3
  data['shock'] = (data['priceavg'].pct_change()) * 100
  data['shock'] = data['shock'].apply(lambda x: 1 if x >= percent else 0)
  data.drop(['high', 'low', 'close'], axis = 1, inplace = True)
  return data


# In[63]:


priceboolean_shock(10, bse_data)


# In[64]:


altdata_pricepercentchg = priceboolean_shock(10, bse_data)
altdata_pricepercentchg.reset_index(inplace=True)
altdata_pricepercentchg.rename(columns={list(altdata_pricepercentchg)[0]:'date'}, inplace=True)


# In[65]:


alt.Chart(altdata_pricepercentchg).mark_point().encode(
    x='date',
    y='priceavg'
)


# In[66]:


plotpricepercentchg = altdata_pricepercentchg.filter(['date', 'priceavg'], axis=1) #df.copy()
plotpricepercentchg.index = pd.Index(sm.tsa.datetools.dates_from_range('2015', length=len(altdata_pricepercentchg['date']))) 
del plotpricepercentchg['date']
sm.graphics.tsa.plot_pacf(plotpricepercentchg.values.squeeze())
plt.show()


# In the partial autocorrelation plot above, we have statistically significant partial autocorrelations at lag values 0, 1, 2, 4, 6, 7, 8, 15, 16, 21, 22, 25, 26. And lag values 0, 1 represents a perfect positive correlation.  While the rest of values are very close to 0 and under the confidence intervals, which are represented as blue shaded regions

# ## Creating the reverse dummy time series:
# 
# Price shocks : we will be creating a 0/1 dummy-coded boolean time series for shock, based on whether closing price at T vs T+1 has a difference > 2%. ( 0/1 dummy-coding is for direction of shock). This will be reverse of pct_change()

# In[67]:


def reverseboolean_shock(percent, df, col):
  data = df.filter(['date', col], axis=1) #df.copy()
  data.set_index('date', inplace=True)
  data = data.reindex(index=data.index[::-1])
  data['percentchg'] = (data[col].pct_change()) * 100
  data['shock'] = data['percentchg'].apply(lambda x: 1 if x > percent else 0)
  data.drop(col, axis = 1, inplace = True)
  data = data.reindex(index=data.index[::-1])
  return data.dropna()


# In[68]:


reverseboolean_shock(2, bse_data, 'close')


# In[69]:


altdata_closepercentchg = reverseboolean_shock(2, bse_data, 'close')
altdata_closepercentchg.reset_index(inplace=True)
altdata_closepercentchg.rename(columns={list(altdata_closepercentchg)[0]:'date'}, inplace=True)


# In[70]:


alt.Chart(altdata_closepercentchg).mark_point().encode(
    x='date',
    y='percentchg'
)


# Pricing black swan : we will be creating a 0/1 dummy-coded boolean time series for shock, based on whether closing price at T vs T+1 has a difference > 5%. ( 0/1 dummy-coding is for direction of shock). This will be reverse of pct_change()

# In[71]:


reverseboolean_shock(5, bse_data, 'close')


# In[72]:


altdata_closepercentchg5 = reverseboolean_shock(5, bse_data, 'close')
altdata_closepercentchg5.reset_index(inplace=True)
altdata_closepercentchg5.rename(columns={list(altdata_closepercentchg5)[0]:'date'}, inplace=True)


# In[73]:


alt.Chart(altdata_closepercentchg5).mark_point().encode(
    x='date',
    y='percentchg'
)


# Pricing shock without volume shock : Now we will be creating a time series for pricing shock without volume shock based on whether price at T vs T+1 has a difference > 2%. ( 0/1 dummy-coding is for direction of shock). This will be reverse of pct_change()
# 
# 

# In[74]:


def pricereverseboolean_shock(percent, df):
  data = df.filter(['date', 'high', 'low','close'], axis=1) #df.copy()
  data.set_index('date', inplace=True)
  data = data.reindex(index=data.index[::-1])
  data['reversepriceavg'] = (data['high'] + data['low'] + data['close']) / 3
  data['shock'] = (data['reversepriceavg'].pct_change()) * 100
  data['shock'] = data['shock'].apply(lambda x: 1 if x >= percent else 0)
  data.drop(['high', 'low', 'close'], axis = 1, inplace = True)
  data = data.reindex(index=data.index[::-1])
  return data.dropna()


# In[75]:


pricereverseboolean_shock(2, bse_data)


# In[76]:


altdata_reversepricepercentchg = pricereverseboolean_shock(2, bse_data)
altdata_reversepricepercentchg.reset_index(inplace=True)
altdata_reversepricepercentchg.rename(columns={list(altdata_reversepricepercentchg)[0]:'date'}, inplace=True)


# In[77]:


alt.Chart(altdata_reversepricepercentchg).mark_point().encode(
    x='date',
    y='reversepriceavg'
)


# # Textual Analysis of news from Times of India News Headlines

# In[78]:


#reading the uploaded csv file and assigning to news variable
news  = pd.read_csv('/kaggle/input/india-press/india-news-headlines.csv')


# In[79]:


#getting the overview of all the columns in the news dataset
news.columns


# In[80]:


#finding the total rows and columns of news dataset
news.shape


# In[81]:


#first 5 rows content of the dataset
news.head()


# In[82]:


#converting publish_date column to 
news['publish_date'] = pd.to_datetime(news['publish_date'], format = '%Y%m%d')


# In[83]:


#first 5 rows content of the dataset
news.head()


# In[84]:


#last 5 rows content of the dataset
news.tail()


# In[85]:


#getting brief overview of the dataset - number of columns and rows (shape of dataset), columns names and its dtype, how many non-null values it has and memory usage.
news.info()


# In[86]:


#finding unique values in headline_category
news['headline_category'].unique()


# In[87]:


#checking all the values count (unique values total count)
news['headline_category'].value_counts()


# In[88]:


#total unique values count
news['headline_category'].value_counts().count()


# In[89]:


#checking all the values count (unique values total count)
news['headline_text'].value_counts()


# In[90]:


#total unique values count
news['headline_text'].value_counts().count()


# In[91]:


#finding if any null values are present
news.isnull().sum().sum()


# In[92]:


#finding if any duplicate values are present
news.duplicated().sum()


# In[93]:


#rough checking by marking all duplicates as True. Default is first which marks duplicates as True except for the first occurrence.
news.duplicated(keep=False).sum()


# In[94]:


#sorting the dataset to delete the duplicates, to make duplicates come together one after another. The sorted dataset index values are also changed
cols = list(news.columns)
news.sort_values(by=cols, inplace=True, ignore_index=True)


# In[95]:


news[news.duplicated(keep=False)]


# In[96]:


#dropping the duplicates only keeping the last value (ordinally last row from sorted) of each duplicates
news.drop_duplicates(keep='last', inplace=True, ignore_index=True)


# In[97]:


#re-checking everything worked well with drop_duplicates() carried out earlier on the dataset
news.duplicated().sum()


# In[98]:


from textblob import TextBlob


# In[99]:


#getting a list of unique dates in publish_date column
lst = news['publish_date'].value_counts().index.tolist()


# In[100]:


#concatenating all the headline_text column values of same date in publish_date column
new = []
for x in lst:
  df = news.loc[news['publish_date'] == x]
  headlinetext = ''
  publishdate = str(x)
  headlinetext = df['headline_text'].iloc[0]
  for i in range(1 , len(df)):
    headlinetext = headlinetext + '. '+ df['headline_text'].iloc[i]  
  new.append(headlinetext)


# In[101]:


#creating a new dataset
newsdf = pd.DataFrame({'publish_date': lst, 'headline_text' : new})


# In[102]:


newsdf


# In[103]:


#sorting the dataset based on dates
newsdf.sort_values(by='publish_date', inplace=True, ignore_index=True)


# In[104]:


newsdf.head()


# In[105]:


newsdf.tail()


# In[106]:


newsdf.info()


# We can calculate the sentiment using TextBlob. Based on the polarity, we determine whether it is a positive text or negative or neutral. For TextBlog, if the polarity is more than 0, it is considered positive, if it is less than 0 then it is considered negative and if it ia=s equal to 0 is considered neutral. Subjectivity quantifies the amount of personal opinion and factual information contained in the text. The higher subjectivity means that the text contains personal opinion rather than factual information. 

# In[107]:


polarity = []
subjectivity = []
for idx, row in newsdf.iterrows():
  polarity.append(TextBlob(row['headline_text']).sentiment[0])
  subjectivity.append(TextBlob(row['headline_text']).sentiment[1])


# In[108]:


newsdf['polarity'] = polarity
newsdf['subjectivity'] = subjectivity


# In[109]:


newsdf.head()


# In[110]:


newsdf.tail()


# In[111]:


#finding if any null values are present
newsdf.isnull().sum().sum()


# In[112]:


#renameing the publish_date to date so it will help us during joining this dataset with bse_data dataset
newsdf.rename(columns={'publish_date': 'date'}, inplace = True)


# In[113]:


#selecting required columns
newsdf = newsdf.filter(['date', 'polarity', 'subjectivity'], axis=1)


# In[114]:


newsdf.shape


# In[115]:


newsdf['date'].duplicated().sum()


# In[116]:


bse_data.shape


# In[117]:


bse_data['date'].duplicated().sum()


# In[118]:


bse_data = pd.merge(bse_data, newsdf, how='left', on=['date'])


# In[119]:


bse_data.shape


# In[120]:


bse_data.head()


# In[121]:


bse_data.tail()


# In[122]:


#finding if any null values are present
bse_data.isnull().sum().sum()


# # Preparing the dataset for machine learning

# In[123]:


#adding new row for 30th June 2020 (price to be predicted of this day) to main dataset to get average values of all the columns for this day
#taking average because we don't know the values of all the columns for tomorrow so to predict we need average for independent variable.
#We will separate this row later from this main dataset so we can use this as prediction of unseen data for tomorrow. 
#And then tally it with actual data from unseenbse_data dataset which we have downloaded too for 30th June 2020 actual values
bse_data.loc[len(bse_data)] = ['2020-06-30', bse_data['open'].mean(), bse_data['high'].mean(), bse_data['low'].mean(),
                       bse_data['close'].mean(), bse_data['adjclose'].mean(), bse_data['volume'].median(), newsdf['polarity'].mean(), newsdf['subjectivity'].mean() ]


# In[124]:


#converting date from object dtype to datetime dtype
bse_data['date'] = pd.to_datetime(bse_data['date'], format="%Y-%m-%d")


# In[125]:


bse_data.tail()


# In[126]:


bse_data["month"] = bse_data['date'].dt.month
bse_data["day"] = bse_data['date'].dt.day
bse_data["dayofweek"] = bse_data['date'].dt.dayofweek
bse_data["week"] = bse_data['date'].dt.week
bse_data['movingavg4weeks'] = round(bse_data['close'].rolling(window=(4*5), min_periods = 1).mean().shift(),2)
bse_data['movingavg16weeks'] = round(bse_data['close'].rolling(window=(16*5), min_periods = 1).mean().shift(),2) #add 12 weeks to 4 weeks 
bse_data['movingavg28weeks'] = round(bse_data['close'].rolling(window=(28*5), min_periods = 1).mean().shift(),2) #add 12 weeks to 16 weeks
bse_data['movingavg40weeks'] = round(bse_data['close'].rolling(window=(40*5), min_periods = 1).mean().shift(),2) #add 12 weeks to 28 weeks
bse_data['movingavg52weeks'] = round(bse_data['close'].rolling(window=(52*5), min_periods = 1).mean().shift(),2)  #add 12 weeks to 40 weeks
bse_data['window10days'] = round(bse_data['close'].rolling(window = 10, min_periods = 1).mean().shift(),2)  
bse_data['window50days'] = round(bse_data['close'].rolling(window = 50, min_periods = 1).mean().shift(),2)
bse_data['volumeshock'] = round(boolean_shock(10, bse_data, 'volume').reset_index()['shock'], 2)
bse_data['closeshock2'] = round(reverseboolean_shock(2, bse_data, 'close').reset_index()['shock'], 2)
bse_data['closeshock5'] = round(reverseboolean_shock(5, bse_data, 'close').reset_index()['shock'],2)
bse_data['closeshock10'] = round(reverseboolean_shock(10, bse_data, 'close').reset_index()['shock'], 2)
bse_data['priceshock'] = round(priceboolean_shock(10, bse_data).reset_index()['shock'], 2)
bse_data['reversebooleanshock2'] = round(reverseboolean_shock(2, bse_data, 'close').reset_index()['shock'], 2)
bse_data['reversebooleanshock5'] = round(reverseboolean_shock(5, bse_data, 'close').reset_index()['shock'], 2)
bse_data['pricereverseshock2'] = round(pricereverseboolean_shock(2, bse_data).reset_index()['shock'], 2)
bse_data['polarity'] = round(bse_data['polarity'] , 2)
bse_data['subjectivity'] = round(bse_data['subjectivity'] , 2)
bse_data['price'] = round((bse_data['high'] + bse_data['low'] + bse_data['close']) / 3 , 2)
bse_data['close'] = round(bse_data['close'] , 2)


# In[127]:


bse_data.columns


# In[128]:


bse_data


# In[129]:


#fillinf the null columns
bse_data.fillna(method = 'bfill', inplace = True)


# In[130]:


#fillinf the null columns
bse_data.fillna(method = 'ffill', inplace = True)


# In[131]:


#finding if any null values are present
bse_data.isnull().sum().sum()


# In[132]:


#selecting specific columns
bse_data = bse_data.filter(['month', 'day', 'dayofweek', 'week',
       'movingavg4weeks', 'movingavg16weeks', 'movingavg28weeks',
       'movingavg40weeks', 'movingavg52weeks', 'window10days', 'window50days',
       'volumeshock', 'closeshock2', 'closeshock5', 'closeshock10',
       'priceshock', 'reversebooleanshock2', 'reversebooleanshock5',
       'pricereverseshock2', 'polarity', 'subjectivity', 'price', 'close'], axis=1)


# In[133]:


bse_data


# In[134]:


#separating the predicted date row from main dataset after getting all the calculated average values
main_bsedata = bse_data.iloc[:1345,:].reset_index()  
newtestunseen_bsedata = bse_data.iloc[1345:,:].reset_index()  


# In[135]:


main_bsedata.shape


# In[136]:


main_bsedata.tail()


# In[137]:


newtestunseen_bsedata.shape


# In[138]:


newtestunseen_bsedata.head()


# # Training the model and predicting the price of tomorrow 30th June 2020

# In[139]:


X = main_bsedata.drop(['price','close'], axis = 1)
y = main_bsedata[['price','close']]


# In[140]:


Xnewtestunseen = newtestunseen_bsedata.drop(['price','close'], axis = 1)
ynewtestunseen_ans = newtestunseen_bsedata[['price','close']]


# In[141]:


X.shape, y.shape


# In[142]:


Xnewtestunseen.shape, ynewtestunseen_ans.shape


# In[143]:


split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# In[144]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# There are two ways to predict values of two columns one is
# 
#   * Direct Multioutput Regression:
#   
#   It involves seperating each target variable as independent regression problem, that is here it presumably assumes the outputs to be independent of each other.
# 
#   * Chained Multioutput Regression:
# 
#   It involves creating a series of regression models from single output regression model, that is the first model in the sequence uses the input and predicts one output then the second model uses the input and the output from the first model to make a prediction and it goes on depending on the number of target variables.
# 
# 
# In this case Chained Multioutput Regression will be more appropriate option as the stock price ((high+low+close)/3) and closing price are interdependent.

# In[145]:


rfg = RandomForestRegressor(random_state = 42, n_estimators = 500, criterion='mse', max_depth = 30, min_samples_leaf=2, min_samples_split=5, n_jobs=1)


# In[146]:


chainedmodel = RegressorChain(rfg)
chainedmodel.fit(X_train, y_train)


# In[147]:


pred = chainedmodel.predict(X_test)
roundpred = []
for x in range(len(pred)):
  roundpred.append([round(pred[x][0], 2),round(pred[x][1], 2) ])


# In[148]:


r2_score(y_test, roundpred) 


# In[149]:


#evaluating the performance of the model
#MAE
print('MAE')
print(mean_absolute_error(y_test, roundpred), end='\n')
#MSE
print('MSE')
print(mean_squared_error(y_test, roundpred), end='\n')
#RMSE
print('RMSE')
print(np.sqrt(mean_squared_error(y_test, roundpred)))


# In[150]:


pred_newtestunseen = chainedmodel.predict(Xnewtestunseen)


# In[151]:


[(round(pred_newtestunseen[0][0], 2)),(round(pred_newtestunseen[0][1], 2))]


# In[152]:


ynewtestunseen_ans  #used average of high, low, close, volume to calculate price ((high+low+close)/3) and close value


# In[153]:


[unseenbsedata_price[0] , round(unseenbse_data['close'],2)[0]] #actual price ((high+low+close)/3) calculated earlier and the closing price


# The model predicted for 30th June 2020 the price ((high+low+close)/3) i.e the average of high, low, close of BSE index to be 35020.02 and closing price to be 34955.46
# 
# And the actual price ((high+low+close)/3) i.e the average of high, low, close of BSE index on day 30th June 2020 was 34987.5, and closing price was 34915.8
# 
# So as seen above our model has done a very good prediction
