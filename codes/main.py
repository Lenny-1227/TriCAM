import math
import random
import sys
from time import time

import numpy as np
import torch
import torch.optim as optim
from Models import MONET
from utility.batch_test import data_generator, test_torch
from utility.parser import parse_args
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, data_config, args):
        self.args = args
        # argument settings
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]

        self.feat_embed_dim = args.feat_embed_dim
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.n_layers = args.n_layers
        self.has_norm = args.has_norm
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.lamb = self.regs[1]
        self.alpha = args.alpha
        self.beta = args.beta
        self.dataset = args.dataset
        self.model_name = args.model_name
        self.agg = args.agg
        self.target_aware = args.target_aware
        self.cf = args.cf
        self.cf_gcn = args.cf_gcn
        self.lightgcn = args.lightgcn
        self.margin=args.margin
        self.temperature=args.temperature
        self.lambda_mmd = args.lambda_mmd

        self.use_contrastive = args.use_contrastive
        self.use_cross_attention = args.use_cross_attention
        self.use_mlp = args.use_mlp

        self.nonzero_idx = data_config["nonzero_idx"]

        self.image_feats = np.load("data/{}/image_feat.npy".format(self.dataset))
        self.text_feats = np.load("data/{}/text_feat.npy".format(self.dataset))

        self.model = MONET(
            self.n_users,
            self.n_items,
            self.feat_embed_dim,
            self.nonzero_idx,
            self.has_norm,
            self.image_feats,
            self.text_feats,
            self.n_layers,
            self.alpha,
            self.beta,
            self.agg,
            self.cf,
            self.cf_gcn,
            self.lightgcn,
            margin=self.margin,
            temperature=self.temperature,
            lambda_mmd=self.lambda_mmd,
            use_contrastive=self.use_contrastive,
            use_cross_attention=self.use_cross_attention,
            use_mlp=self.use_mlp,
        )

        self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

        self.train_losses = []  # Initialize for tracking training loss
        self.val_losses = []    # Initialize for tracking validation loss
        self.precision_values = []
        self.recall_values = []
        self.ndcg_values = []
        self.hit_ratio_values = []

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, prediction, _, _, _, _ = self.model()
        result = test_torch(
            ua_embeddings,
            ia_embeddings,
            users_to_test,
            is_val,
            self.adj,
            self.beta,
            self.target_aware,
        )
        return result, prediction, _, _
    

    def train(self):
        nonzero_idx = torch.tensor(self.nonzero_idx).cuda().long().T
        self.adj = (
            torch.sparse_coo_tensor(
                nonzero_idx,
                torch.ones((nonzero_idx.size(1))).cuda(),
                (self.n_users, self.n_items),
            )
            .to_dense()
            .cuda()
        )
        stopping_step = 0

        n_batch = data_generator.n_train // self.args.batch_size + 1
        best_recall = 0
        for epoch in range(self.args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0.0, 0.0, 0.0, 0.0
            n_batch = data_generator.n_train // self.args.batch_size + 1
            for _ in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()
                users, pos_items, neg_items = data_generator.sample()
                user_emb, item_emb, prediction, _, _, _, mmd_loss = self.model(pos_items=pos_items, neg_items=neg_items, users=users)
               
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.model.bpr_loss(
                    user_emb, item_emb, users, pos_items, neg_items, self.target_aware
                )

                batch_emb_loss = self.decay * batch_emb_loss
                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                if mmd_loss is not None:
                    batch_loss += self.args.lambda_mmd * mmd_loss

                batch_loss.backward(retain_graph=True)
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

                del user_emb, item_emb, prediction
                torch.cuda.empty_cache()

            self.lr_scheduler.step()

            if math.isnan(loss):
                print("ERROR: loss is nan.")
                sys.exit()

            perf_str = "Pre_Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]" % (
                epoch,
                time() - t1,
                loss,
                mf_loss,
                emb_loss,
                reg_loss,
            )
            print(perf_str)

            self.train_losses.append(loss)

            if epoch % self.args.verbose != 0:
                continue

            t2 = time()
            users_to_val = list(data_generator.val_set.keys())
            ret, _, _, _ = self.test(users_to_val, is_val=True)

            # Compute validation loss
            val_loss, val_mf_loss, val_emb_loss, val_reg_loss = 0.0, 0.0, 0.0, 0.0
            with torch.no_grad():
                self.model.eval()
                ua_embeddings, ia_embeddings, prediction, _, _, _, mmd_loss = self.model(pos_items=pos_items, neg_items=neg_items, users=users)
                for user in users_to_val:
                    pos_items = data_generator.val_set[user]
                    neg_items = random.sample(
                        range(self.n_items), len(pos_items)
                    )  # Randomly sample negatives for validation

                    # Compute BPR loss
                    batch_mf_loss, batch_emb_loss, batch_reg_loss = self.model.bpr_loss(
                        ua_embeddings, ia_embeddings, [user], pos_items, neg_items, self.target_aware
                    )
                    batch_emb_loss = self.decay * batch_emb_loss
                    batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                    if mmd_loss is not None:
                        batch_loss += self.args.lambda_mmd * mmd_loss

                    val_loss += float(batch_loss)
                    val_mf_loss += float(batch_mf_loss)
                    val_emb_loss += float(batch_emb_loss)
                    val_reg_loss += float(batch_reg_loss)

            # Average validation losses
            num_val_users = len(users_to_val)
            val_loss /= num_val_users
            val_mf_loss /= num_val_users
            val_emb_loss /= num_val_users
            val_reg_loss /= num_val_users

            self.val_losses.append(val_loss)
            
            t3 = time()

            if self.args.verbose > 0:
                perf_str = (
                    "Pre_Epoch %d [%.1fs + %.1fs]: val==[%.5f=%.5f + %.5f + %.5f], "
                    "recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]"
                    % (
                        epoch,
                        t2 - t1,
                        t3 - t2,
                        val_loss,
                        val_mf_loss,
                        val_emb_loss,
                        val_reg_loss,
                        ret["recall"][0],
                        ret["recall"][-1],
                        ret["precision"][0],
                        ret["precision"][-1],
                        ret["hit_ratio"][0],
                        ret["hit_ratio"][-1],
                        ret["ndcg"][0],
                        ret["ndcg"][-1],
                    )
                )
                print(perf_str)

            if ret["ndcg"][1] > best_recall:
                best_recall = ret["ndcg"][1]
                stopping_step = 0
                torch.save(
                    {self.model_name: self.model.state_dict()},
                    "./models/" + self.dataset + "_" + self.model_name,
                )
            elif stopping_step < self.args.early_stopping_patience:
                stopping_step += 1
                print("#####Early stopping steps: %d #####" % stopping_step)
            else:
                print("#####Early stop! #####")
                break
    

        self.model = MONET(
            self.n_users,
            self.n_items,
            self.feat_embed_dim,
            self.nonzero_idx,
            self.has_norm,
            self.image_feats,
            self.text_feats,
            self.n_layers,
            self.alpha,
            self.beta,
            self.agg,
            self.cf,
            self.cf_gcn,
            self.lightgcn,
            margin=self.margin,
            temperature=self.temperature,
            lambda_mmd=self.lambda_mmd,
            use_contrastive=self.use_contrastive,
            use_cross_attention=self.use_cross_attention,
            use_mlp=self.use_mlp,
        ).cuda()

        self.model.load_state_dict(
            torch.load(
                "./models/" + self.dataset + "_" + self.model_name,
                map_location=torch.device("cpu")
            )[self.model_name]
        )
        self.model.cuda()
        test_ret = self.test(list(data_generator.test_set.keys()), is_val=False)
        print("Final ", test_ret)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu


if __name__ == "__main__":
    args = parse_args(True)
    set_seed(args.seed)

    config = dict()
    config["n_users"] = data_generator.n_users
    config["n_items"] = data_generator.n_items

    nonzero_idx = data_generator.nonzero_idx()
    config["nonzero_idx"] = nonzero_idx

    trainer = Trainer(config, args)
    trainer.train()
