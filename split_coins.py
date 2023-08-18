#read train.csv and save every coin in its own csv file using Asset_ID as the name of the file
import pandas as pd


#train.csv is to big, I will read it in chunks
#and save every coin in its own csv file using Asset_ID as the name of the file

#read train.csv in chunks
chunksize = 10 ** 6
for chunk in pd.read_csv('train.csv', chunksize=chunksize):

    #for every chunk split it by Asset_ID, if file already exists append to it
    for name, group in chunk.groupby('Asset_ID'):
        try:
            group.to_csv('{}.csv'.format(name), mode='a', header=False)
        except:
            group.to_csv('{}.csv'.format(name), mode='w', header=True)

