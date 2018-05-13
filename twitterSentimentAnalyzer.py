from nltk.sentiment.vader import SentimentIntensityAnalyzer
from IPython import get_ipython
from TwitterAPI import TwitterAPI, TwitterPager

import matplotlib.pyplot as plt
import os

# setup
searchHashtag = os.environ['SEARCH_HASHTAG']

twitterApi = TwitterAPI(os.environ['TWITTER_CONSUMER_KEY'], os.environ['TWITTER_CONSUMER_SECRET'],
                        os.environ['TWITTER_ACCESS_TOKEN'], os.environ['TWITTER_ACCESS_TOKEN_SECRET'])


sentimentIntensityAnalyzer = SentimentIntensityAnalyzer()

get_ipython().run_line_magic('matplotlib', 'osx')
plt.ion()
fig = plt.figure()
plt.axis([0, 1, 0, 1])

twitterPager = TwitterPager(twitterApi, 'search/tweets', {'q': searchHashtag, 'lang': 'en', 'tweet_mode': 'extended'})
for tweet in twitterPager.get_iterator():
    polarityScores = sentimentIntensityAnalyzer.polarity_scores(tweet['full_text'])
    plt.scatter(polarityScores['pos'], polarityScores['neg'], s=2)
    plt.show()
    plt.pause(0.01)
