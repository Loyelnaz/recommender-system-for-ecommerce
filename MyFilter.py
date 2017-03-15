import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from utils import topPredictions
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis = 1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:,np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis = 1)]).T
    elif type == 'item' :
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis = 1)])
    return pred

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

def func(row):
    return int(row['user_id'].split(",")[0])


header = ['item_id', 'user_id', 'score']
df = pd.read_csv('./OnlineMod.csv', sep = "\t", names = header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

print "Users: ", n_users, "Items: ", n_items

train_data, test_data = cv.train_test_split(df, test_size = 0.25)

train_data_matrix = np.zeros((n_items, n_users))

for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_items, n_users))

for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

user_similarity = pairwise_distances(train_data_matrix, metric = 'cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric = 'cosine')

item_prediction = predict(train_data_matrix, item_similarity, type = 'item')
user_prediction = predict(train_data_matrix, user_similarity, type = 'user')
print user_prediction
print "User-based error: ", rmse(user_prediction, test_data_matrix)
print "Item-based error: ", rmse(item_prediction, test_data_matrix)

# topPredictions(item_prediction)
topPredictions(user_prediction)

#---------------Model Based CF-----------#

# sparsity = round(1.0-len(df)/float(n_users*n_items),3)
# print "Sparsity in dataset: ", sparsity*100, "%"
#
# u, s, vt = svds(train_data_matrix, k = 20)
# s_diag_matrix = np.diag(s)
# x_pred = np.dot(np.dot(u, s_diag_matrix), vt)
# # topPredictions(x_pred)
# print "User Based error in Model: ", rmse(x_pred, test_data_matrix)
