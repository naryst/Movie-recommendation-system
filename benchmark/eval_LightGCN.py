import pandas as pd
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from sklearn.metrics import mean_squared_error
import numpy as np
import sklearn.preprocessing as pp
import torch
from torch import nn
from sklearn.model_selection import train_test_split

data_cols = ["user_id", "item_id", "rating", "timestamp"]
test_data = pd.read_csv("ua.test", sep="\t", names=data_cols)
test_data.drop(columns="timestamp", axis=1, inplace=True)

# load train dataset to encode the labels in test data in correct way
data = pd.read_csv("../data/interim/ratings.csv", index_col=0)
data = data[data["rating"] >= 3]
train, _ = train_test_split(data.values, test_size=0.2, random_state=16)
train_data = pd.DataFrame(train, columns=data.columns)
le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()
train_data["user_id_idx"] = le_user.fit_transform(train_data["user_id"].values)
train_data["item_id_idx"] = le_item.fit_transform(train_data["item_id"].values)
train_user_ids = train_data["user_id"].unique()
train_item_ids = train_data["item_id"].unique()
test_data = test_data[
    (test_data["user_id"].isin(train_user_ids))
    & (test_data["item_id"].isin(train_item_ids))
]
test_data["user_id_idx"] = le_user.transform(test_data["user_id"].values)
test_data["item_id_idx"] = le_item.transform(test_data["item_id"].values)


def data_loader(data, batch_size, n_usr, n_itm):
    # Function to sample negative items not interacted with
    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    # Grouping interactions by user and creating a DataFrame
    interected_items_df = (
        data.groupby("user_id_idx")["item_id_idx"].apply(list).reset_index()
    )
    indices = [x for x in range(n_usr)]

    # Sampling users for the batch
    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
    users.sort()
    users_df = pd.DataFrame(users, columns=["users"])

    # Merging user interactions with the selected batch of users
    interected_items_df = pd.merge(
        interected_items_df,
        users_df,
        how="right",
        left_on="user_id_idx",
        right_on="users",
    )

    # Selecting positive items randomly from user interactions
    pos_items = (
        interected_items_df["item_id_idx"].apply(lambda x: random.choice(x)).values
    )

    # Generating negative items for each user
    neg_items = interected_items_df["item_id_idx"].apply(lambda x: sample_neg(x)).values

    # Returning tensors for users, positive items, and negative items
    return (
        torch.LongTensor(list(users)).to(device),
        torch.LongTensor(list(pos_items)).to(device) + n_usr,
        torch.LongTensor(list(neg_items)).to(device) + n_usr,
    )


class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr="add")

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        # Norm factor for message passing layer from the paper
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class RecSysGNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers,
        num_users,
        num_items,
    ):
        super(RecSysGNN, self).__init__()

        self.embedding = nn.Embedding(num_users + num_items, latent_dim)

        self.convs = nn.ModuleList(LightGCNConv() for _ in range(num_layers))

    def forward(self, edge_index):
        emb0 = self.embedding.weight
        embs = [emb0]

        emb = emb0
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        out = torch.mean(torch.stack(embs, dim=0), dim=0)  # alpha = 1/(K+1)

        return emb0, out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, out = self(edge_index)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items],
        )


# make tensor values be from 1 to 5
def rescale_tensor(tensor):
    min_ = tensor.min()
    max_ = tensor.max()
    return 1 + (tensor - min_) / (max_ - min_) * 4


def calculate_rmse(test_data, predicted_ratings):
    true_ranks = []
    pred_ranks = []
    for row in test_data.iterrows():
        row = row[1]
        cur_user = row["user_id_idx"]
        cur_item = row["item_id_idx"]
        true_ranks.append(row["rating"])
        pred_ranks.append(predicted_ratings[cur_user][cur_item].item())
    mse = mean_squared_error(true_ranks, pred_ranks)
    return np.sqrt(mse)


def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, test_data, K):
    test_user_ids = torch.LongTensor(test_data["user_id_idx"].unique())

    # compute the score of all user-item pairs
    relevance_score = torch.matmul(
        user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1)
    )

    # scale relevance score to make it represent user scores
    predicted_scores = rescale_tensor(relevance_score).detach().cpu()
    # calculate RMSE for test part of the dataset
    # It's use rating difference on true rating given by user
    # and rating predicted by the model
    rmse = calculate_rmse(test_data, predicted_scores)

    # compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(
        topk_relevance_indices.cpu().numpy(),
        columns=["top_indx_" + str(x + 1) for x in range(K)],
    )

    topk_relevance_indices_df["user_ID"] = topk_relevance_indices_df.index
    topk_relevance_indices_df["top_relevant_item"] = topk_relevance_indices_df[
        ["top_indx_" + str(x + 1) for x in range(K)]
    ].values.tolist()

    topk_relevance_indices_df = topk_relevance_indices_df[
        ["user_ID", "top_relevant_item"]
    ]

    # measure overlap between recommended (top-scoring) and held-out user-item
    # interactions
    test_interacted_items = (
        test_data.groupby("user_id_idx")["item_id_idx"].apply(list).reset_index()
    )

    metrics_df = pd.merge(
        test_interacted_items,
        topk_relevance_indices_df,
        how="left",
        left_on="user_id_idx",
        right_on=["user_ID"],
    )
    metrics_df["intrsecting_item"] = [
        list(set(a).intersection(b))
        for a, b in zip(metrics_df.item_id_idx, metrics_df.top_relevant_item)
    ]

    metrics_df["recall"] = metrics_df.apply(
        lambda x: len(x["intrsecting_item"]) / len(x["item_id_idx"]), axis=1
    )
    metrics_df["precision"] = metrics_df.apply(
        lambda x: len(x["intrsecting_item"]) / K, axis=1
    )
    return metrics_df["recall"].mean(), metrics_df["precision"].mean(), rmse


latent_dim = 128
n_layers = 5

n_users = 943
n_items = 1546

EPOCHS = 100
BATCH_SIZE = 512
LAMBDA = 1e-4
LR = 0.005
K = 20

lightgcn = RecSysGNN(
    latent_dim=latent_dim,
    num_layers=n_layers,
    num_users=n_users,
    num_items=n_items,
)

lightgcn.load_state_dict(torch.load("../models/GCN.pt"))
lightgcn.to("cuda")
train_edge_index = torch.load("../models/edges.pt")

lightgcn.eval()
with torch.no_grad():
    _, out = lightgcn(train_edge_index)
    final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
    test_topK_recall, test_topK_precision, rmse = get_metrics(
        final_user_Embed, final_item_Embed, n_users, n_items, test_data, K
    )

print("Evaluation results on the test set:")
print(f"Racall - {test_topK_recall}")
print(f"Precision - {test_topK_precision}")
print(f"RMSE - {rmse}")
