import os 
import glob
import pandas as pd
os.chdir("/Users/dannysibley/desktop/Housing_Price_ML")

#credited:
#https://stackoverflow.com/questions/9234560/find-all-csv-files-in-a-directory-using-python/12280052

extension= 'csv'
all_filenames = [i for i in glob.glob('*{}'.format(extension))]

#combine all files in list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

#export to csv, saved as combined_csv.csv
combined_csv.to_csv('combined_csv.csv', index = False, encoding = 'utf-8-sig')