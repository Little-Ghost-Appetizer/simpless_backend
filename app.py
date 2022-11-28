from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from URLSearchParams import URLSearchParams
#import urllib.parse as urlparse
#from urllib.parse import urlencode
import requests
#from threading import Timer
#import time
#from utils import generate_tags
import threading
import pandas as pd
import numpy as np
import re
import nltk
import math
nltk.download('stopwords', quiet=True)
nltk.download('wordnet')
nltk.download('omw-1.4')

from string import punctuation
from string import digits
from langdetect import detect_langs
from nltk.stem import WordNetLemmatizer
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

def preprocess_df(data, NUM_USERS=5, NUM_TWEETS=5):
    print("Reading data...")

    df = pd.DataFrame(data)
    #print(df.head())

    # Sort by date and remove duplicate tweets
    df = df.sort_values(by=['date'], ascending=False)
    df = df.drop_duplicates(subset=['url'], keep='first')

    # Log Importance
    #print(df.info())
    df['importance'] = df['likes'] + df['retweets'] + df['comments_num'] + df['quotes_num']
    df['tweet_cnt'] = df.groupby('user_account')['importance'].transform('sum')
    df['importance'] = df['importance'].apply(log_importance).apply(int)
    df['importance'] = df['importance'].replace(0, 1)
    df = df.reset_index(drop=True)
    
    popular_users = extract_users(df, NUM_USERS)
    popular_tweets = extract_tweets(df, NUM_TWEETS)
    
    df['clean_content'] = df['content'].str.repeat(df['importance'])

    # Extract tags
    df['clean_content'] = df['clean_content'].apply(remove_return)
    df['tags'] = df['clean_content'].apply(extract_tags)
    
    # Filter out non-English tweets
    df['lang'] = df['clean_content'].apply(find_language)
    df = df[df['lang'] == 'en']
    
    # Clean text
    print("Cleaning text...")
    df['clean_content'] = df['clean_content'].apply(clean_text)
    df = df.reset_index(drop=True)
    
    return df, popular_users, popular_tweets

def extract_users(df, NUM_USERS=5):
    # Return df with columns: user_account, profile_image_url
    popular_users = df.sort_values(by=['tweet_cnt'], ascending=False)
    popular_users = popular_users.drop_duplicates(subset=['user_account'], keep='first')
    popular_users = popular_users[['user_account', 'profile_image_url']]
    popular_users = popular_users.reset_index(drop=True).head(min(popular_users.shape[0], NUM_USERS))
    return popular_users

def extract_tweets(df, NUM_TWEETS=5):
    # Return df with columns: url, user_account, profile_image_url, content
    popular_tweets = df.sort_values(by=['importance'], ascending=False)
    popular_tweets = popular_tweets[['url', 'user_account', 'profile_image_url', 'content']]
    popular_tweets = popular_tweets.reset_index(drop=True).head(min(popular_tweets.shape[0], NUM_TWEETS))
    return popular_tweets


### Lemmatization tool
stemmer = WordNetLemmatizer()

def remove_return(x):
    return re.sub(r"\n", " ", x)

def clean_text(x):
    x = re.sub(r"&amp;", " ", x)
    x = re.sub(r"@\S*", "", x)
    x = re.sub(r"https://\S*", "", x)
    x = re.sub(r"#\S*", "", x)
    
    # Remove punctuation
    x = ''.join(ch if ch not in set(punctuation) else " " for ch in x)
    
    # Remove all single characters
    x = re.sub(r'\W', ' ', x)
    x = re.sub(r'\s+[a-zA-Z]\s+', ' ', x)
    x = re.sub(r'\^[a-zA-Z]\s+', ' ', x)
    
    # Substituting multiple spaces with single space
    x = re.sub(r'\s+', ' ', x, flags=re.I)
    
    # Removing all numbers
    x = x.translate(str.maketrans('', '', digits))
    
    # Converting to Lowercase
    x = x.lower()
    
    # Lemmatization and remove stopwords
    x = x.split()
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [stemmer.lemmatize(word) for word in x if word not in stopwords]
    x = ' '.join(tokens)
    
    return x

def log_importance(x):
    return np.log2(x) if x > 0 else 0

def extract_tags(x):
    tags = []
    tokens = x.split(" ")
    for token in tokens:
        if token and token[0] == "#":
            tags.append(token.lower()[1:] if token[-1].isalnum() else token[1:-1])
    return tags

def find_language(x):
    try:
        return str(detect_langs(x)[0])[:2]
    except:
        return 'None'

def tfidf_embed(documents):
    # documents: list of str
    tfidfv = TfidfVectorizer().fit(documents)
    return tfidfv.transform(documents).toarray(), tfidfv.vocabulary_

def collect_words(df, top_n=10):
    if df.empty:
        return []
    arr, vocab = tfidf_embed(df['clean_content'])
    vocab = dict((v,k) for k,v in vocab.items())

    scores = arr.sum(axis=0) # array([5.05080122, 3.8244604 , 3.41681198, ...]) score for each word

    words = []
    top_n = min(top_n, len(scores))
    for idx in np.argsort(-scores)[:top_n]:
        # print(vocab[idx], scores[idx])
        words.append(vocab[idx])
    return words

def collect_tags(df, num_tags=10):
    tags = []
    for lst in df['tags'].tolist():
        if lst:
            tags += lst
    # print(len(tags))
    c = Counter(tags) # [('#itaewon', 562), ('#ateez', 193), ('#itaewontragedy', 140), ...]
    return list(map(lambda x: '#'+x[0], c.most_common(num_tags)))

def generate_tags(data, num_tags):
    df, popular_users, popular_tweets = preprocess_df(data, NUM_USERS=5, NUM_TWEETS=5)
    tags = collect_tags(df, num_tags)
    words = collect_words(df, num_tags)
    return {'keywords': tags + words,
            'posts': popular_tweets.to_dict('records'),
            'users': popular_users.to_dict('records')}

wait_exist = 0
sem = threading.Semaphore()
wait = threading.Semaphore()

#tum = "555"
app = Flask(__name__)
CORS(app)

val_to_key_str = {
    "value1": "url",
    "value2": "keyword",
    "value3": "username",
    "value4": "user_id",
    "value5": "user_account",
    "value6": "profile_image_url",
    "value7": "content",
    "value8": "likes",
    "value9": "retweets",
    "value10": "comments_num",
    "value11": "quotes_num",
    "value12": "media_urls",
    "value13": "image_urls",
    "value14": "date",
    "value15": "sentiment_scores"
}

data = []
next_token = ''
scrape_count = 0
scrape_round = 0
search = []
search_idx = 0
upper_round = 3
upper_count = 10

class Simpless:
    #def __init__(self):
    #    self.scrape_count = 0
    def scrape(self, searchParams, headers):
        global wait_exist, data, next_token, scrape_count, scrape_round, search
        print("Scrape Round: ",scrape_round)
        print("Scrape Count: ",scrape_count)
        print("Token: ",next_token)
        scrape_count += 1
        passs = False
        while not passs:
            response = None
            try:
                if wait_exist:
                    next_token = ''
                    search.clear()
                    sem.release()
                    abort(500)
                if next_token:
                    print("Shooting URL: ", URLSearchParams(searchParams).append({"next_token": next_token}))
                    response = requests.get(URLSearchParams(searchParams).append({"next_token": next_token}),headers=headers)
                else:
                    print("Shooting URL: ", searchParams)
                    response = requests.get(searchParams,headers=headers)

            except requests.exceptions.Timeout:
                print("Timeout")
                continue
            except requests.exceptions.TooManyRedirects:
                print("TooManyRedirects")
                continue
            except requests.exceptions.RequestException as e:
                print("Other error")
                continue
            else:
                if response.status_code != 200:
                    print("Weird situation happen")
                    print(response.status_code)
                    print(response.text)
                    return -1
                #print(response)
                #print(response.text)
                json = response.json()
                temp_data = json["data"]
                if len(temp_data) == 0:
                    return 0
                passs = True
                temp_data = [{val_to_key_str[k]: v for k,v in d.items()} for d in temp_data]
                new_temp_data = []
                for d in temp_data:
                    new_t_d = {}
                    for k,v in d.items():
                        if k == "likes" or k == "retweets" or k == "quotes_num" or k == "comments_num":
                            new_t_d[k] = int(v)
                        else:
                            new_t_d[k] = v
                    new_temp_data.append(new_t_d)
                data.extend(new_temp_data)
                next_token = json["next_token"]
        return 1
    def process(self):
        global wait_exist, data, next_token, scrape_count, scrape_round, search, search_idx, upper_round, upper_count
        print("Search: ", search)
        print("Start_round: ", scrape_round)
        wait_exist += 1
        print("wait_exist: ", wait_exist)
        sem.acquire()
        wait_exist -= 1
        if wait_exist:
            next_token = ''
            search.clear()
            sem.release()
            abort(500)
        #request_json = request.json
        request_json = request.args
        print("req: ", request_json)
        print("continue: ", request_json["continue"])
        print("search_keyword: ", request_json["search_keyword"])

        if  (not next_token) or (not "continue" in request_json) or (request_json["continue"]=='false'):
            #search = request_json["search_keyword"]
            search = request.args.getlist("search_keyword")
            search_idx = 0
            next_token = ""
            if "upper_round" in request_json:
                upper_round = math.ceil(int(request_json["upper_round"])/len(search))
            else:
                upper_round = 3
            if "upper_count" in request_json:
                upper_count = int(request_json["upper_count"])
            else:
                upper_count = 10
            data.clear()
            scrape_count = 0
            scrape_round = 0
        if search_idx >= len(search):
            num_tags = 10
            if "num_tags" in request_json:
                num_tags = int(request_json["num_tags"])
            if len(data) == 0:
                return jsonify({"finished": True})
            sem.release()
            try:
                ml_result = generate_tags(data, num_tags)
                ml_result["finished"] = True
                return jsonify(ml_result)
            except:
                return jsonify({"finished": True})

        src = URLSearchParams("http://api.hashscraper.com/api/twitter?")
        searchParams = src.append({
            "api_key": API_KEY,
            "keyword": search[search_idx],
            "max_count": 5,
        })
        headers = {'Content-Type': 'application/json; version=2'}

        if search_idx < len(search) and scrape_round < upper_round:
            while scrape_count < upper_count:
                scrape_result = self.scrape(searchParams,headers)
                if not scrape_result:
                    search_idx += 1
                    scrape_round = 0
                    scrape_count = 0
                    sem.release()
                    if search_idx>=len(search) and len(data) == 0:
                        return jsonify({"finished": True})
                    else:
                        return jsonify({"keywords":[], "finished": False})
                elif scrape_result == -1:
                    sem.release()
                    return jsonify({"keywords":[], "finished": False})
                scrape_count += 1
            scrape_count = 0
            scrape_round += 1
            if scrape_round >= upper_round:
                search_idx += 1
                scrape_round = 0
            sem.release()
            return jsonify({"keywords":[], "finished": False})
        else:
            """
            next_token = ""
            search = ""
            scrape_round = 0
            """
            num_tags = 10
            if "num_tags" in request_json:
                num_tags = int(request_json["num_tags"])
            if len(data) == 0:
                return jsonify({"finished": True})
            sem.release()
            try:
                ml_result = generate_tags(data, num_tags)
                ml_result["finished"] = True
                return jsonify(ml_result)
            except:
                return jsonify({"finished": True})
            #time.sleep(10)
        #for k,v in self.data[0].items():
        #    print(k, type(v), v)

        #return jsonify(self.data)
"""
@app.route('/api/simplesssearch', methods=['POST'])
def simplesssearch():
   simp = Simpless()
   return simp.process()
"""
@app.route('/api/simplesssearch', methods=['GET'])
def simplesssearch():
   simp = Simpless()
   return simp.process()

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.route('/shutdown',methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if __name__ == '__main__':
    context = ('server.crt', 'server.key')
    response = requests.get("https://api.ipify.org")
    print(response.text)
    #app.run(debug=True, port=5000)
    #app.run(host='0.0.0.0', ssl_context=context)
    app.run(debug=True, host='0.0.0.0')
