import sys
import numpy as np
import pandas as pd
import sklearn.linear_model

# Load the data
df = pd.read_csv('data/Advertising.csv')

#Given a dataframe, df, and title of x column and y column,
#computes linear regression formula
def linearReg(df, x, y):
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

#Given a dataframe, df, and title of y column, computes multiple linear regression formula
def multipleLinearReg(df, y):
    yvals = df[y].tolist() #Grab result values
    xvals = df.drop(y, axis=1) #remove Y values from x
    matrix = xvals.values #turn x into a matrix
    new_matrix = np.ones((matrix.shape[0], matrix.shape[1] + 1)) #Add first column of ones for intercept term
    new_matrix[:, 1:] = matrix #Fill new matrix with old matrix
    transpose = new_matrix.transpose() #create transpose matrix
    y_matrix = np.array(yvals)  #turn y into a matrix
    beta = np.linalg.inv(transpose @ new_matrix) @ transpose @ y_matrix #optimize error function to find beta
    return beta

def polynomialLinearReg(df, x, y, degree):
    xvals = df[x]
    # x_matrix = np.array(xvals) //numpy way to do it
    # y_matrix = np.array(yvals)
    # beta = np.polyfit(x_matrix, y_matrix, degree)
    for i in range(2, degree+1):
        df["x"+str(i)] = xvals**i
    return multipleLinearReg(df, y)


# print(LinearReg(df, 'TV', 'sales'))
df = df.drop(df.columns[0], axis=1)
print(df)
print(multipleLinearReg(df, 'sales'))