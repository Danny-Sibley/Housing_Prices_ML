"""
Plan for project
find large amount of housing data online within last couple years in local area
potentially create sql database with data to incorportate sql as well
clean data and remove anamolies
model data using python libraries 
use several different ML models to make predictions
compare models to zillow and red fin estimates
potentailly create aggregate model that takes all three into account
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from seaborn import regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn import ensemble


# imports and reads data from csv file
data = pd.read_csv('combined_csv.csv')


# removes unwanted features
df2 = data.drop(['SALE TYPE', 'SOLD DATE', 'PROPERTY TYPE', 'ADDRESS', 'CITY', 'STATE OR PROVINCE', 'LOCATION', 'DAYS ON MARKET',
                'HOA/MONTH', 'STATUS', 'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME', 'URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)', 'SOURCE', 'MLS#', 'FAVORITE', 'INTERESTED',
                 'LATITUDE', 'LONGITUDE', '$/SQUARE FEET'], axis=1)

# renames columns, in same matrix
df2.rename(columns={'ZIP OR POSTAL CODE': 'ZIPCODE', 'SQUARE FEET': 'SQFT',
           'LOT SIZE': 'LOT_SIZE', 'YEAR BUILT': 'YEAR_BUILT', '$/SQUARE FEET': '$/SQFT'}, inplace=True)

# data cleaning
# remove entry rows that have incomplete data
df3 = df2.dropna()
# print(df3.describe())

# sorts data by sqft to view outliers
df3 = df3.drop(labels=[2152, 2596], axis=0)
#print(df3.sort_values(by = 'SQFT'))

# sorts data by beds to view outliers, removing one with 14 beds but only 1705 sqft
df3 = df3.drop(labels=2481, axis=0)
# print(df3.sort_values(by='BEDS'))

# sorts data by price to view outliers
df3 = df3.drop(labels=[585, 1726, 595, 87, 415, 89, 429, 1043], axis=0)
#print(df3.sort_values(by = 'PRICE'))

# saves cleaned data frame as csv file
# df3.to_csv('cleaned_housing_data.csv')

# store dependent variable seperatly
prices = df3['PRICE']
# remaining variables placed in features
features = df3.drop('PRICE', axis=1)
# print(features)


# displays some info about cleaned data
# print(df3.info())
# rint(df3.describe())

# sets pandas option to display all columns
pd.set_option('display.max_columns', None)


# #Bar chart to show number of bedrooms
# df3['BEDS'].value_counts().plot(kind='bar')
# plt.title('number of Bedroom')
# plt.xlabel('Bedrooms')
# plt.ylabel('Count')
# sns.despine
# plt.show()

# shows latititude and longitude
# plt.figure(figsize=(10,10))
# sns.jointplot(x=df2.Longitude.values, y=df2.Latitude.values, height=10)
# plt.ylabel('Latitude', fontsize=12)
# plt.xlabel('Longitude', fontsize=12)
# #plt.show()
# #plt1 = plt()
# sns.despine

# plots price vs sqft
# plt.scatter(df3.SQFT,df3.PRICE)
# plt.xlabel("SQFT")
# plt.ylabel("Price")
# plt.title("Price vs Square Feet")
# plt.show()

# plots BEDS vs PRICE
# plt.scatter(df3.BEDS,df3.PRICE,)
# plt.xlabel("BEDS")
# plt.ylabel("PRICE")
# plt.title("Price vs Square Feet")
# plt.show()

# plots price vs BATHS
# plt.scatter(df3.BATHS,df3.PRICE)
# plt.xlabel("BATHS")
# plt.ylabel("PRICE")
# plt.title("BATHS vs PRICE")
# plt.show()


# plot shows zipcode vs price
# plt.scatter(df3.ZIPCODE,df3.PRICE)
# plt.xlabel("Zip_Code")
# plt.ylabel('Price')
# plt.title("Zip_Code vs Price")
# plt.show()

# plots lot size vs price
# plt.scatter(df3.LOT_SIZE,df3.PRICE)
# plt.xlabel("Lot Size")
# plt.ylabel('Price')
# plt.title("lot size vs Price")
# plt.show()

# plots year built vs price
# plt.scatter(df3.YEAR_BUILT,df3.PRICE)
# plt.xlabel("Year_Built")
# plt.ylabel('Price')
# plt.title("Year_Built vs Price")
# plt.show()

# creates scatter matrix that shows correlations between data
# scatter_matrix(df3)
# plt.show()

# distribution plot of prices
# sns.displot(prices)
# plt.title('Sale Price Distribution')
# plt.xlabel('Price')
# plt.show()

# correlation matrix, numbers closer to 1 indicate more of a correlation
# corrMatrix = df3.corr()
# sns.heatmap(corrMatrix, annot=True)
# plt.show()

# creates instance of Linear regression class
reg = LinearRegression()

# shuffle and split data
X_train, X_test, y_train, y_test = train_test_split(
    features, prices, test_size=0.1, random_state=32)

reg.fit(X_train.values, y_train)

# Tests other linear regression models
# 2. Ridge

ridge = Ridge(alpha=0.5)
ridge.fit(X_train.values, y_train)
ridge_yhat = ridge.predict(X_test.values)

# 3. Lasso

lasso = Lasso(alpha=0.01)
lasso.fit(X_train.values, y_train)
lasso_yhat = lasso.predict(X_test.values)

# 4. Bayesian

bayesian = BayesianRidge()
bayesian.fit(X_train.values, y_train)
bayesian_yhat = bayesian.predict(X_test.values)

# 5. ElasticNet

en = ElasticNet(alpha=0.01)
en.fit(X_train.values, y_train)
en_yhat = en.predict(X_test.values)


print(f'R-Squared of Ridge model is {r2_score(y_test, ridge_yhat)}')
print(f'R-Squared of Lasso model is {r2_score(y_test, lasso_yhat)}')
print(f'R-Squared of Bayesian model is {r2_score(y_test, bayesian_yhat)}')
print(f'R-Squared of ElasticNet is {r2_score(y_test, en_yhat)}')


# makes prediction
# define one new data instance
# ZIPCODE, BEDS, BATHS, SQFT, LOT_SIZE, YEAR_BUILT
Xnew = [[97003, 4, 2, 1800, 6000, 1980]]
# make a prediction
reg_predict = reg.predict(Xnew)
# show the inputs and predicted outputs
print(f' Zipcode: {Xnew[0][0]}, BEDS: {Xnew[0][1]}, BATHS: {Xnew[0][2]}, SQFT: {Xnew[0][3]}, LOT_SIZE: {Xnew[0][4]}, YEAR_BUILT: {Xnew[0][5]} Predicted home price is: ${reg_predict}')


# checks score of regression fit
reg_score = reg.score(X_test.values, y_test)
print(f' reg score: {reg_score}')

# gradient boosting
clf = ensemble.GradientBoostingRegressor(
    n_estimators=500, max_depth=3, min_samples_split=5, learning_rate=0.2, loss='huber')
clf.fit(X_train.values, y_train.values)

# makes clf prediction
clf_predict = clf.predict(Xnew)
print(f' CLF prediction: ${clf_predict}')

# checks clf score
clf_score = clf.score(X_test.values, y_test)
print(f'clf score: {clf_score}')

