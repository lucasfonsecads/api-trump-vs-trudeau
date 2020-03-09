#Set seed for reprucibility
import random

random.seed(53)

# Import all we need from sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
import pandas as pd
import os

def detect_tweet(data):
    # Load data
    with open(os.path.join(os.path.dirname(__file__), 'tweets.csv')) as f: 
        tweet_df = pd.read_csv(f)

    # Create target
    y = tweet_df.author

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(tweet_df['status'], y, test_size=.33, random_state=53)

    # Initialize count vectorizer
    count_vectorizer = CountVectorizer(stop_words="english", min_df=.05, max_df=.9)

    # Create count train and test variables
    count_train = count_vectorizer.fit_transform(X_train)
    count_test = count_vectorizer.transform(X_test)

    # Initialize tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=.05, max_df=.9)

    # Create tfidf train and test variables
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Create a MulitnomialNB model
    tfidf_nb = MultinomialNB()

    # ... Train your model here ...
    tfidf_nb.fit(tfidf_train, y_train)

    # Run predict on your TF-IDF test data to get your predictions
    tfidf_nb_pred = tfidf_nb.predict(tfidf_test)

    # Calculate the accuracy of your predictions
    tfidf_nb_score = metrics.accuracy_score(y_test, tfidf_nb_pred)

    # Create a MulitnomialNB model
    count_nb = MultinomialNB()

    # ... Train your model here ...
    count_nb.fit(count_train, y_train)

    # Run predict on your count test data to get your predictions
    count_nb_pred = count_nb.predict(count_test)

    # Calculate the accuracy of your predictions
    count_nb_score = metrics.accuracy_score(y_test, count_nb_pred)

    # Create a LinearSVM model
    tfidf_svc = LinearSVC()

    # ... Train your model here ...
    tfidf_svc.fit(tfidf_train, y_train)

    # Run predict on your tfidf test data to get your predictions
    tfidf_svc_pred = tfidf_svc.predict(tfidf_test)

    # Calculate your accuracy using the metrics module
    tfidf_svc_score = metrics.accuracy_score(y_test, tfidf_svc_pred)


    tweet = data['tweet']
    user_choice = data['user_choice']
    
    
    if user_choice == "Donald J. Trump":
        trump_tweet = tweet
        trump_tweet_vectorized = tfidf_vectorizer.transform([trump_tweet])
        trump_tweet_pred = tfidf_svc.predict(trump_tweet_vectorized)
        check_user_answer = trump_tweet_pred.item(0)

        if check_user_answer == user_choice:

            response = {
                'message': "Predicted Trump tweet: {}".format(trump_tweet_pred.item(0)),
                'linear_svc_score': "%0.3f" % tfidf_svc_score,
                'user_answer': True,
                'machine_answer': True
            }
            
            return response
        else:

            response = {
                'message': "Predicted Trump tweet: {}".format(trump_tweet_pred.item(0)),
                'linear_svc_score': "%0.3f" % tfidf_svc_score,
                'user_answer': False,
                'machine_answe': True
            }
            
            return response    
    else:

        trudeau_tweet = tweet    
        trudeau_tweet_vectorized = tfidf_vectorizer.transform([trudeau_tweet])
        trudeau_tweet_pred = tfidf_svc.predict(trudeau_tweet_vectorized)
        check_user_answer = trudeau_tweet_pred.item(0)

        if check_user_answer == user_choice: 

            response = {
                'message': "Predicted Trudeau tweet: {}".format(trudeau_tweet_pred.item(0)),
                'linear_svc_score': "%0.3f" % tfidf_svc_score,
                'user_answer': True,
                'machine_answer': True
            }
            
            return response
        else:

            response = {
                'message': "Predicted Trudeau tweet: {}".format(trudeau_tweet_pred.item(0)),
                'linear_svc_score': "%0.3f" % tfidf_svc_score,
                'user_answer': False,
                'machine_answer': True
            }
            
            return response


    return
