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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# imports and reads data from csv file
data = pd.read_csv('combined_csv.csv')

# renames columns
#data.columns = ['Sale Type', 'W', 'X', 'Y', 'Z']
# creates copy with new column names
df2 = data.set_axis(['Sale_type', 'Sold_Date', 'Property_Type', 'Address', 'City', 'State', 'Zip_Code', 'Price', 'BEDS', 'BATHS', 'Location', 'SQFT', 'Lot_Size', 'Year_Built', 'Days_On_Market',
                    '$/SQFT', 'HOA/Month', 'Status', 'Next_OPen_house', 'Next_Open_house_end', 'URL', 'Source', 'MLS', 'Favorite', 'Interested', 'Latitude', 'Longitude'], axis=1, inplace=False)

prices = df2['Price']
#removes unwanted features
keptfeatures = df2.drop(['Sale_type','Property_Type','Address','City','State','Location','Days_On_Market','HOA/Month','Status','Next_OPen_house', 'Next_Open_house_end', 'URL', 'Source', 'MLS', 'Favorite', 'Interested'], axis =1)

# sets pandas option to display all columns
pd.set_option('display.max_columns', None)

# displays some key data metrics
# print(df2.describe())

# print(data.head())
# print(data.describe())

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

# plots prive vs sqft
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

# displays info about data
# df2.info()
features = ['Sold_Date', 'City', 'Zip_Code', 'Price', 'BEDS',
            'BATHS', 'SQFT', 'Latitude', 'Longitude', 'Year_Built']
# scatter_matrix(df2[features])
# plt.show()

# correlation matrix
corrMatrix = df2.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

#defining performance metric
def peformance_metric(y_true, y_predict):
    #calculates and returns performance score between true and predicted based on metric chosen
    score = r2_score(y_true, y_predict)
    return score 
    
#shuffle and split data
X_train, X_test, y_train, y_test = train_test_split(keptfeatures, prices, test_size= 0.2, random_state= 42)


