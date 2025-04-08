import pandas as pd
import pickle
import numpy as np

class ModelHelper():
    def __init__(self):
        # model
        self.model = pickle.load(open("model.pkl", 'rb'))

    def predictions(self, grams, quality, successful_transactions, rating, product_title_sentiment):
        
        df = pd.DataFrame()
        df["grams"] = [grams]
        df["quality"] = [quality]
        df["successful_transactions"] = [successful_transactions]
        df["rating"] = [rating]
        df["product_title_sentiment"] = [product_title_sentiment]
       

        # columns in order
        df = df.loc[:, ['grams', 'quality', 'successful_transactions', 'rating',
        'product_title_sentiment']]
        
        preds = self.model.predict(df)
        return(preds[0][1])