import pickle
import pandas as pd
import numpy as np
import logging
import spacy
import re



try:

    df = pd.read_csv("notebooks/data/sample30.csv")
    df =df.dropna(axis=0,subset=['reviews_username','reviews_rating'])
    df=df.drop_duplicates(subset=['reviews_username'])
    with open("pickle_files/item_correlation.pkl","rb") as f:
        item_correlation = pickle.load(f)
    item_pivot=df.pivot(index='reviews_username',columns='id',values='reviews_rating').fillna(0).T
    mean = np.nanmean(item_pivot,axis=1)
    df_subtracted = item_pivot - mean[:,None]
    item_predicted_rating = np.dot(df_subtracted.T,item_correlation)
    dummy = df.copy()
    dummy.reviews_rating = dummy.reviews_rating.apply(lambda x:0 if x>=1 else 0)
    dummy= dummy.pivot(index='reviews_username',columns='id',values='reviews_rating').fillna(1)

    final_rating = np.multiply(item_predicted_rating,dummy)
    with open("pickle_files/model.pkl","rb") as f:
        model=pickle.load(f)
    with open("notebooks/tf_idf.pkl","rb") as f:
        tf_idf_vec=pickle.load(f)
    nlp = spacy.load('en_core_web_sm')
except Exception as e:

    logging.error(f"Exception occurred {e}")

def pre_process(text):
    text = text.lower()
    text = re.sub('\d|\!|\|:|\?|\.|,|-','',text)
    lemma=" ".join([token.lemma_ for token in nlp(text) if not token.is_stop])
    return lemma

class RecommendationModel:


    """
    This function uses item-item correlation matrix to predict the top_n recommended products for the user
    """
    def top_n_recommendations(self,userName,top_n=5):
        try:
           
            return final_rating.loc[userName].sort_values(ascending=False)[0:top_n].index
        except KeyError as e:
            logging.error("Invalid username",e)
            raise e
   
   

    """
    This function takes list of items recommended as input to predict the top_n products with positive review.
    It uses the sentiment prediction model to predict the sentiment of each review for an item.
    Top_n items with higher number of positive reviews are returned.
    """
    def top_n_positive_recommendations(self,recommendation_list,top_n=5):
            sentiment_review={}
            for item in recommendation_list:
                reviews=df[df['id']==item].reviews_text
                name = df[df['id']==item].name.iloc[0]
                category = df[df['id']==item].categories.iloc[0]
                # pre-process text
                logging.info("pre-processing review text")
                reviews=reviews.apply(pre_process)
                logging.info(f"predicting sentiment for {item}")
                X=tf_idf_vec.transform(reviews).toarray()
                y =model.predict(X)
                    # count all 1 s
                sentiment_review[name]=len([x for x in y if x==1])
            # sort dictionary values
            return dict(sorted(sentiment_review.items(),key=lambda x:x[1],reverse=True)[0:top_n]).keys()
    
    
