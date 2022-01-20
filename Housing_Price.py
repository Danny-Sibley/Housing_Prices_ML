"""
Plan for project
find large amount of housing data online within last couple years in local area
potentially find sql database with data to incorportate sql as well
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
import mpl_toolkits
from pandas.plotting import scatter_matrix
from seaborn import regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn import ensemble


# imports and reads data from csv file
data = pd.read_csv('combined_csv.csv')


# removes unwanted features
df2 = data.drop(['SALE TYPE', 'SOLD DATE', 'PROPERTY TYPE', 'ADDRESS', 'CITY', 'STATE OR PROVINCE', 'LOCATION', 'DAYS ON MARKET',
                'HOA/MONTH', 'STATUS', 'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME', 'URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)', 'SOURCE', 'MLS#', 'FAVORITE', 'INTERESTED',
                 'LATITUDE', 'LONGITUDE'], axis=1)               
# renames columns, in same matrix
df2.rename(columns={'ZIP OR POSTAL CODE': 'ZIPCODE'}, inplace=True)

# remove entry rows that have incomplete data
df3 = df2.dropna()
#print(df3.describe())

# sorts data by price to view outliers 
#print(df3.sort_values(by = 'PRICE'))

# sorts data by beds to view outliers, removing one with 14 beds but only 1705 sqft
print(df3.sort_values(by = 'BEDS'))
df3= df3.drop(labels= 2481, axis=0)
print(df3.sort_values(by = 'BEDS'))

# store dependent variable seperatly
prices = df3['PRICE']

# remaining variables placed in features
features = df3.drop('PRICE', axis=1)


# displays some info about cleaned data
# print(df3.info())
# print(df3.describe())


# removed sold date to remove error with fits
#keptfeatures = df3.drop(['Sale_type','Sold_Date', 'Price','Property_Type','Address','City','State','Location','Days_On_Market','HOA/Month','Status','Next_OPen_house', 'Next_Open_house_end', 'URL', 'Source', 'MLS', 'Favorite', 'Interested'], axis =1)


# sets pandas option to display all columns
pd.set_option('display.max_columns', None)


# #Bar chart to show number of bedrooms
# df2['BEDS'].value_counts().plot(kind='bar')
# plt.title('number of Bedroom')
# plt.xlabel('Bedrooms')
# plt.ylabel('Count')
# sns.despine
# #plt.show()

# shows latititude and longitude
# plt.figure(figsize=(10,10))
# sns.jointplot(x=df2.Longitude.values, y=df2.Latitude.values, height=10)
# plt.ylabel('Latitude', fontsize=12)
# plt.xlabel('Longitude', fontsize=12)
# #plt.show()
# #plt1 = plt()
# sns.despine

# plots price vs sqft
# plt.scatter(df2.Price,df2.SQFT)
# plt.title("Price vs Square Feet")

# plt.scatter(df2.Price,df2.Longitude)
#plt.title("Price vs Location of the area")

# plt.scatter(df2.Zip_Code,df2.Price)
# plt.xlabel("Zip_Code")
# plt.ylabel('Price')
# plt.title("Zip_Code vs Price")

# plt.scatter(df2.Year_Built,df2.Price)
# plt.xlabel("Year_Built")
# plt.ylabel('Price')
# plt.title("Year_Built vs Price")

# shows graph of zip code vs price
# plt.scatter(df3.ZIPCODE, prices)
# plt.title('Price vs ZIPCODE')
# plt.show()

# creates scatter matrix that shows correlations between data
# scatter_matrix(df3)
# plt.show()

# distribution plot of prices
sns.displot(prices)
plt.title('Sale Price Distribution')
plt.xlabel('Price')
#plt.show()

# correlation matrix
corrMatrix = df3.corr()
sns.heatmap(corrMatrix, annot=True)
# plt.show()

# creates instance of Linear regression class
reg = LinearRegression()

# shuffle and split data
X_train, X_test, y_train, y_test = train_test_split(
    features, prices, test_size=0.2, random_state=42)

reg.fit(X_train, y_train)

# checks score of regression fit
reg_score = reg.score(X_test, y_test)
print(f' reg score: {reg_score}')

# gradient boosting
clf = ensemble.GradientBoostingRegressor(
    n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='squared_error')
clf.fit(X_train, y_train)

clf_score = clf.score(X_test, y_test)
print(f'clf score: {clf_score}')


#t_sc = np.zeros((params['n_estimators']),dtype=np.float64)
# original_params = {‘n_estimators’: 400, ‘max_depth’: 5,’min_samples_split’: 2,’random_state’: 2,’learning_rate’: 0.1,‘loss’:’ls’}
#params = dict(original_params)

# defining performance metric
def performance_metric(y_true, y_predict):
    # calculates and returns performance score between true and predicted based on metric chosen
    score = r2_score(y_true, y_predict)
    return score


"""
def fit_model(X, y):
    # performs grid search over max depth for decision tree regressor trained on input data X,y

    # create cross validation sets from training data
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    # create decision tree regressor
    regressor = DecisionTreeRegressor()

    # create dictionary for parameter max depth
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    # transform performance metric into a scoring function using make scorer
    scoring_fnc = make_scorer(performance_metric)

    # grid search cv object -> gridsearchCV()
    grid = GridSearchCV(estimator=regressor, param_grid=params,
                        scoring=scoring_fnc, cv=cv_sets)

    # fit grid search object to the data to compute optimal model
    grid = grid.fit(X, y)

    # return the optimal model to fit the data
    return grid.best_estimator_


# fit training data to model using grid search
reg = fit_model(X_train, y_train)

# produce value for max depth
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

"""
