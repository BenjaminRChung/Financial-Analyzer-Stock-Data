from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

''' Getting Article Data from Finviz using Request and Beautiful Soup to parse HTML Code '''

#RAW URL
finviz_url = 'https://finviz.com/quote.ashx?t='

#Common Tickers - Stocks
tickers = ['AMZN', 'GOOG', 'FB']

news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
     
    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

''' Parsing and Manipulating Finviz Data - Found the ticker (stock), date, time, and title '''

parsed_data = []

for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.split(' ')

        #If title contains date + time and not just time
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        
        parsed_data.append([ticker, date, time, title])


'''Sentiment Analysis - breaks down a message into topic chunks and assigns a sentiment score to each topic. Polarity Scores include negative, neutral, and positive. The sum of scores is represented as a compound score - ranging from -1 to +1.
AI bots trained on millions of pieces of text, combining Natural Language Processing (NLP) and Machine Learning (ML). Applies sentiment analysis to Title.'''

#Using Pandas to assign data to particular columns/rows
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

#Initialize Vader as part of the nltk module - sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()

#Getting just the compound score from the Sentiment analyzer, adding a new column to my Pandas dataframe with the compound score only 
compound_score = lambda title: vader.polarity_scores(title)['compound']
df['compound score'] = df['title'].apply(compound_score)
df['date'] = pd.to_datetime(df.date).dt.date


'''Visualizing Parsed Data - using Matplotlib'''

plt.figure(figsize=(10,8))

#Calculates average of compound score for articles released in a day (groups dayS)
mean_df = df.groupby(['ticker', 'date']).mean()

#Rearranging the compound score in the Pandas dataframe making it easily plotable 
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound score', axis="columns").transpose()
mean_df.plot(kind='bar')

#Plotting Pandas Dataframe
plt.show()