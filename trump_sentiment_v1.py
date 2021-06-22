import tweepy
import pandas as pd
import numpy as np
import json

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob
import re

#%matplotlib inline

# Función que importa los Tokens de un archivo JSON
def twitter_tokens():
    with open('twitter_tokens.json') as json_file:
        tokens = json.load(json_file)
        return tokens

# Función que autentica para el uso de la API
def twitter_setup(tokens_twitter):    
    auth = tweepy.OAuthHandler(tokens_twitter["CONSUMER_KEY"],tokens_twitter["CONSUMER_SECRET"])    
    auth.set_access_token(tokens_twitter["ACCESS_TOKEN"], tokens_twitter["ACCESS_TOKEN_SECRET"])
    api = tweepy.API(auth)
    return api

extractor = twitter_setup(twitter_tokens())
tweets = extractor.user_timeline(screen_name = "Colombia", count = 200)
print("5 recent tweets: \n")
for tweet in tweets[:5]:
    print(tweet.text)
    print()

# Comprensión de listas para obtener un atributo del objeto, apendizarlos y
# almacenarlos en la variable data con el nombre columna de  "Tweets"
data = pd.DataFrame(data = [tweet.text for tweet in tweets], columns = ["Tweets"])
display(data.head(10))

print(dir(tweets[0]))

print(tweets[0].id)
print(tweets[0].created_at)
print(tweets[0].source)
print(tweets[0].favorite_count)
print(tweets[0].retweet_count)
print(tweets[0].geo)
print(tweets[0].coordinates)
print(tweets[0].entities)

# We add relevant data:
data['len']  = np.array([len(tweet.text) for tweet in tweets])
data['ID']   = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])

# We extract the mean of lenghts:
mean = np.mean(data['len'])
print(f"The lenght's average in tweets: {mean}")

# We extract the tweet with more FAVs and more RTs:
fav_max = np.max(data['Likes'])
rt_max  = np.max(data['RTs'])

fav = data[data.Likes == fav_max].index[0]
rt  = data[data.RTs == rt_max].index[0]

# Max FAVs:
print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))

# Max RTs:
print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print(f"Number of retweets: {rt_max}")
print(f"{data['len'][rt]} characters.\n")

# We create time series for data:
tlen = pd.Series(data=data['len'].values, index=data['Date'])
tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])

# Lenghts along time:
tlen.plot(figsize=(16,4), color='r');

# Likes vs retweets visualization:
tfav.plot(figsize=(16,4), label="Likes", legend=True)
tret.plot(figsize=(16,4), label="Retweets", legend=True);

# We obtain all possible sources, Uniques values:
sources = []
for source in data['Source']:
    if source not in sources:
        sources.append(source)

# We print sources list:
print("Creation of content sources:")
for source in sources:
    print("* {}".format(source))
    
# We create a numpy vector mapped to labels:
percent = np.zeros(len(sources))

for source in data['Source']:
    for index in range(len(sources)):
        if source == sources[index]:
            percent[index] += 1
            pass

percent /= 100

# Pie chart:
pie_chart = pd.Series(percent, index=sources, name='Sources')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6));


# =============================================================================
# Sentiment Analysis
# =============================================================================
def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
    
# We create a column with the result of the analysis:
data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])

# We display the updated dataframe with the new column:
display(data.head(10))

# We construct lists with classified tweets:
pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]
    
# We print percentages:
print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print("Percentage de negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))
