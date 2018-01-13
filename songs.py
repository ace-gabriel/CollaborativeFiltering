import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


# Simple Item-Based Collaborative Filtering

data = pd.read_csv('data.csv')
# In IBCF we don't care about user
item_data = data.drop('user', 1)
# Create an Item vs. Item table to store cosine similarities
data_ifs = pd.DataFrame(index = item_data.columns, columns = item_data.columns)
# Cosine Sim: sum product of the 1st and 2nd column, divide by the product of the
# Larger the value, more similar the items are
# sqrt of the sum of squares of each column
for i in range(0, len(data_ifs.columns)):
    for j in range(0, len(data_ifs.columns)):
        item_data.ix[i,j] = 1 - cosine(item_data.ix[:,i], item_data.ix[:,j])

# Create Neighbourhood Table
neigh = pd.DataFrame(index = data_ifs.columns, columns = range(1,11))
# Loop through similarity table and find the cloest neighbourhood
for i in range(0, len(data_ifs.columns)):
    neigh.ix[i,:10] = data_ifs.ix[0:,i].sort_values(ascending = False)[:10].index


# Simple User-Based Collaborative Filtering
# Check which items the user has consumed
# For each item being consumed, get top neighbourhood
# Get the consumption record of the user for each neighbour 
# Calculate a similarity score
# Recommend the item with the highest score

def getScore(history, similarities):
    return sum(history * similarities) / sum(similarities)

data_sims = pd.DataFrame(index = data.index, columns = data.columns)
data_sims.ix[:,:1] = data.ix[:,:1]

for i in range(0,len(data_sims.index)):
    for j in range(1,len(data_sims.columns)):
        user = data_sims.index[i]
        product = data_sims.columns[j]

        if data.ix[i][j] == 1:
            data_sims.ix[i][j] = 0
        else:
            product_top_names = neigh.ix[product][1:10]
            product_top_sims = data_ifs.ix[product].sort_values(ascending=False)[1:10]
            user_purchases = item_data.ix[user,product_top_names]

            data_sims.ix[i][j] = getScore(user_purchases,product_top_sims)

# Get the top songs
data_recommend = pd.DataFrame(index=data_sims.index, columns=['user','1','2','3','4','5','6'])
data_recommend.ix[0:,0] = data_sims.ix[:,0]

# Instead of top song scores, we want to see names
for i in range(0,len(data_sims.index)):
    data_recommend.ix[i,1:] = data_sims.ix[i,:].sort_values(ascending=False).ix[1:7,].index.transpose()

# Print a sample
print(data_recommend.ix[:10,:4])
