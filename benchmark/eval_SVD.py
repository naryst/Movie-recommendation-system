import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from scipy.linalg import sqrtm
from sklearn.metrics import mean_squared_error as mse

data_cols = ["user_id", "item_id", "rating", "timestamp"]
test_data = pd.read_csv("ua.test", sep="\t", names=data_cols)
test_data.drop(columns="timestamp", axis=1, inplace=True)
svd_matrix = np.load("../models/SVD.npy")

t = test_data.copy()
true_movies_test = t.groupby("user_id")["item_id"].apply(list).reset_index()

K = 100
intersection = 0
for row in true_movies_test.iterrows():
    user_id = row[1]["user_id"]
    true = row[1]["item_id"]
    user_scores = svd_matrix[user_id - 1]
    pred = user_scores.argsort()[-K:]
    true_set = set(true)
    pred_set = set(pred)
    intersection += len(true_set.intersection(pred_set))
mean_intersection = intersection / (len(true_movies_test))

preds = []
for row in test_data.iterrows():
    user_id = row[1]["user_id"]
    item_id = row[1]["item_id"]
    true_score = row[1]["rating"]
    if user_id <= svd_matrix.shape[0] and item_id <= svd_matrix.shape[1]:
        pred_score = svd_matrix[user_id - 1][item_id - 1]
    else:
        pred_score = 3.5
    preds.append(pred_score)

rmse = np.sqrt(mse(test_data.rating.tolist(), preds))
rmse

print(f"Mean intersection - {mean_intersection}")
print(f"RMSE - {rmse}")
