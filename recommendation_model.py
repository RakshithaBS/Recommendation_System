import pickle
import pandas as pd
import numpy as np
import logging

try:
    df = pd.read_csv("sample30.csv")
    df =df.dropna(axis=0,subset=['reviews_username','reviews_rating'])
    df=df.drop_duplicates(subset=['reviews_username'])
    with open("item_correlation.pkl","rb") as f:
        item_correlation = pickle.load(f)
    item_pivot=df.pivot(index='reviews_username',columns='id',values='reviews_rating').fillna(0).T
    mean = np.nanmean(item_pivot,axis=1)
    df_subtracted = item_pivot - mean[:,None]
    item_predicted_rating = np.dot(df_subtracted.T,item_correlation)
    dummy = df.copy()
    dummy.reviews_rating = dummy.reviews_rating.apply(lambda x:0 if x>=1 else 0)
    dummy= dummy.pivot(index='reviews_username',columns='id',values='reviews_rating').fillna(1)

    final_rating = np.multiply(item_predicted_rating,dummy)
    with open("model.pkl","rb") as f:
        model=pickle.load(f)
    with open("tf_idf_model.pkl","rb") as f:
        tf_idf_vec=pickle.load(f)
    
except Exception as e:

    logging.error("Exception occurred")

class RecommendationModel:


    def top_n_recommendations(self,userName,top_n=5):
        try:
           
            return final_rating.loc[userName].sort_values(ascending=False)[0:top_n].index
        except KeyError as e:
            logging.error("Invalid username",e)
            raise e
   

    def top_n_positive_recommendations(self,recommendation_list,top_n=5):
            sentiment_review={}
            for item in recommendation_list:
                reviews=df[df['id']==item].reviews_text
                name = df[df['id']==item].name.iloc[0]
                category = df[df['id']==item].categories.iloc[0]
                #description=f"name: {name} , category {category}"
                print(f"predicting sentiment for {item}")
                X=tf_idf_vec.transform(reviews).toarray()
                y =model.predict(X)
                    # count all 1 s
                sentiment_review[name]=len([x for x in y if x==1])
            # sort dictionary values
            return dict(sorted(sentiment_review.items(),key=lambda x:x[1],reverse=True)[0:top_n]).keys()