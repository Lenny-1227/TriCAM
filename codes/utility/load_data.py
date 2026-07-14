import json
import os

# args = parse_args()
import random as rd

# from utility.parser import parse_args
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, models


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path + "/5-core"
        self.batch_size = batch_size

        train_file = path + "/5-core/train.json"
        val_file = path + "/5-core/val.json"
        test_file = path + "/5-core/test.json"

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test, self.n_val = 0, 0, 0
        self.neg_pools = {}

        self.exist_users = []

        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))
        for uid, items in train.items():
            if len(items) == 0:
                continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except Exception:
                continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except Exception:
                continue

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set, self.val_set = {}, {}, {}
        for uid, train_items in train.items():
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for _, i in enumerate(train_items):
                self.R[uid, i] = 1.0

            self.train_items[uid] = train_items

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except Exception:
                continue

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except Exception:
                continue
                
        # Load test and validation sets
        self.test_set = {int(uid): items for uid, items in test.items() if items}
        self.val_set = {int(uid): items for uid, items in val.items() if items}

        # Load combined text and image features
        self.text_features = np.load(path + "/text_feat.npy")
        self.image_features = np.load(path + "/image_feat.npy")

    def nonzero_idx(self):
        r, c = self.R.nonzero()
        idx = list(zip(r, c))
        return idx

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items

    def print_statistics(self):
        print("n_users=%d, n_items=%d" % (self.n_users, self.n_items))
        print("n_interactions=%d" % (self.n_train + self.n_val + self.n_test))
        print(
            "n_train=%d, n_val=%d, n_test=%d, sparsity=%.5f"
            % (
                self.n_train,
                self.n_val,
                self.n_test,
                (self.n_train + self.n_val + self.n_test)
                / (self.n_users * self.n_items),
            )
        )


def dataset_merge_and_split(path):
    df = pd.read_csv(path + "/train.csv", index_col=None, usecols=None)
    # Construct matrix
    ui = defaultdict(list)
    for _, row in df.iterrows():
        user, item = int(row["userID"]), int(row["itemID"])
        ui[user].append(item)

    df = pd.read_csv(path + "/test.csv", index_col=None, usecols=None)
    for _, row in df.iterrows():
        user, item = int(row["userID"]), int(row["itemID"])
        ui[user].append(item)

    train_json = {}
    val_json = {}
    test_json = {}
    for u, items in ui.items():
        if len(items) < 10:
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        test = testval[: len(testval) // 2]
        val = testval[len(testval) // 2 :]
        train = [i for i in list(range(len(items))) if i not in testval]
        train_json[u] = [items[idx] for idx in train]
        val_json[u] = [items[idx] for idx in val.tolist()]
        test_json[u] = [items[idx] for idx in test.tolist()]

    with open(path + "/5-core/train.json", "w") as f:
        json.dump(train_json, f)
    with open(path + "/5-core/val.json", "w") as f:
        json.dump(val_json, f)
    with open(path + "/5-core/test.json", "w") as f:
        json.dump(test_json, f)


def load_textual_image_features(data_path):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    asin_dict = json.load(open(os.path.join(data_path, "asin_sample.json"), "r"))

    # Load the SentenceTransformer model
    word_embedding_model = models.Transformer("/home/ps/codes/AAAA_sdb2/@@/MONET/codes/data/stsb-roberta-large")
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    bert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print("SentenceTransformer manually model loaded successfully.")

    # Confirm the model works with a sample text
    sentence = "This is a test sentence."
    embedding = bert_model.encode(sentence)
    print("Embedding shape: {embedding.shape}")
    print(embedding)

    # Prepare textual feture data.
    doc2vec_model = Doc2Vec.load(os.path.join(data_path, "doc2vecFile"))

    # Load visual vectors
    vis_vec = np.load(os.path.join(data_path, "image_feature.npy"), allow_pickle=True).item()
    text_vec = {}
    for asin in asin_dict:
        text_vec[asin] = doc2vec_model.docvecs[asin]

    # Prepare for combined features
    all_dict = {}
    num_items = 0

    # Load train and test datasets
    filename = data_path + "/train.csv"
    df = pd.read_csv(filename, index_col=None, usecols=None)
    for _, row in df.iterrows():
        asin, i = row["asin"], int(row["itemID"])
        all_dict[i] = asin
        num_items = max(num_items, i)
    filename = data_path + "/test.csv"
    df = pd.read_csv(filename, index_col=None, usecols=None)
    for _, row in df.iterrows():
        asin, i = row["asin"], int(row["itemID"])
        all_dict[i] = asin
        num_items = max(num_items, i)

    combined_text_features = []
    v_features = []

   # Extract and combine Doc2Vec and SBERT embeddings
    for i in range(num_items + 1):
        asin = all_dict.get(i)
        if asin in asin_dict:
            doc2vec_emb = doc2vec_model.docvecs[asin]
            sbert_emb = bert_model.encode([asin])[0]

            # Normalize embeddings
            normalized_sbert = normalize(sbert_emb.reshape(1, -1))[0]
            normalized_doc2vec = normalize(doc2vec_emb.reshape(1, -1))[0]

            # Weighted concatenation of embeddings
            combined_emb = np.concatenate((0.7 * normalized_sbert, 0.3 * normalized_doc2vec))
        else:
            combined_emb = np.zeros(doc2vec_model.vector_size + sbert_emb.size)

        combined_text_features.append(combined_emb)
        v_features.append(vis_vec.get(asin, np.zeros(4096)))

    # Save combined text and image features
    np.save(data_path + "/text_feat.npy", np.asarray(combined_text_features, dtype=np.float32))
    np.save(data_path + "/image_feat.npy", np.asarray(v_features, dtype=np.float32))
