import pandas as pd
import pickle
import numpy as np

class ModelHelper():
    def __init__(self):
        # model
        self.model = pickle.load(open("model_fixed.pkl", 'rb'))

    def predictions(self, grams, quality, successful_transactions, rating, product_title_sentiment, ships_to, ships_from):
        sentiment_mapping = {"positive": 1, "negative": 0}
        
        df = pd.DataFrame()
        df["grams"] = [grams]
        df["quality"] = [quality]
        df["successful_transactions"] = [successful_transactions]
        df["rating"] = [rating]
        df["product_title_sentiment"] = [product_title_sentiment]
        df["product_title_sentiment"] = df["product_title_sentiment"].apply(lambda x: sentiment_mapping[x])
        
        df["ships_to"] = [ships_to]
        df["ships_from"] = [ships_from]
       

        # columns in order
        df = df.loc[:, ['grams', 'quality', 'successful_transactions', 'rating',
        'product_title_sentiment', 'ships_to', 'ships_from']]
        
        print(df)
        preds = self.model.predict(df)
        return(preds[0])