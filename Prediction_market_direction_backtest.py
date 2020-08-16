import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import BayesianRidge
from sklearn import lda
from sklearn import qda
import numpy as np


def initialize(context):
   
    context.security = sid(8554) # SPY
    context.window_length = 600
    
    # context.classifier = LogisticRegression(random_state=105) # 5f2f170521785a46dc9b30d7
    context.classifier = lda.LDA() # 5f2f182bfdbbac46b9b29beb
    # context.classifier = qda.QDA() # 5f2f190e6d5d3a473278c6e6
    
    
    context.recent_prices =[]
    context.X = []
    context.Y = []
    
    context.prediction = 0
    
   
    # Rebalance every day, 1 hour after market open.
    schedule_function(
        rebalance,
        date_rules.every_day(),
        time_rules.market_open(hours=1),
    )

    # Record tracking variables at the end of each day.
    schedule_function(
        record_vars,
        date_rules.every_day(),
        time_rules.market_close(minutes = 5),
    )
    
  

def rebalance(context, data):
    context.recent_prices = data.history(context.security, fields ="price", bar_count = context.window_length + 1, frequency = "1d")
    rtns = context.recent_prices.pct_change()
    context.df = pd.concat([rtns.shift(1), rtns.shift(2), rtns], axis=1).dropna()
    context.df.columns = ['lag_1', 'lag_2','Y']
    context.X = context.df[['lag_1', 'lag_2']]
    context.Y = context.df['Y'] > 0
    
    context.classifier.fit(context.X, context.Y)
    
    comb = pd.concat([rtns.shift(1), rtns.shift(2)], axis=1)
    pred = comb.iloc[-1]
    context.prediction = context.classifier.predict(pred)
    
    order_target_percent(context.security, float(context.prediction))
    
    

def record_vars(context, data):
    record(leverage=context.account.leverage)
    pass