import sys
import numpy
import pandas as pd
import requests


df = pd.read_csv('data/Advertising.csv')

#Given a dataframe, df, and title of x column and y column,
#computes linear regression formula
def LinearReg(df, x, y):
#df['cost'] = df['TV'] + df['radio'] + df['newspaper']
#df = df.sort_values(by='cost', ascending=True)
    xbar = df[x].mean()
    ybar = df[y].mean()
    b1Top = 0
    b1Bot = 0
    for index, example in df.iterrows():
        b1Top += (example[x]-xbar)*(example[y]-ybar)
        b1Bot += (example[x] - xbar)**2
    b1 = b1Top/b1Bot
    b0 = ybar - b1*xbar
    return str("y = "+str(b1)+"x + "+str(b0))
#print(LinearReg(df, 'TV', 'sales'))
