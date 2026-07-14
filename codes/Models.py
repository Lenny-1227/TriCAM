import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_geometric.nn.inits import uniform
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def normalize_laplacian(edge_index, edge_weight):
    num_nodes = maybe_num_nodes(edge_index)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_weight

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, temperature=0.07):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, pos_items, users, preferences, pos_emb, neg_emb):
        # Ensure that users are within valid bounds before creating the tensor
        users = torch.tensor(users, dtype=torch.long)

        # Clamp 'users' to valid indices
        users = torch.clamp(users, min=0, max=preferences.size(0) - 1)

        # Move 'users' to the same device as preferences
        users = users.to(preferences.device)

        # Clamp pos_items to be within valid bounds for pos_emb
        pos_items = torch.clamp(pos_items, min=0, max=pos_emb.size(0) - 1)

        # Assert that pos_items are within valid bounds for pos_emb
        assert torch.all(pos_items >= 0) and torch.all(pos_items < pos_emb.size(0)), \
            f"Invalid pos_items indices. Max index: {pos_items.max()}, size: {pos_emb.size(0)}"

        pos_item_emb = pos_emb[pos_items]

        # Positive distance (how far the user preference is from the positive item embedding)
        positive_distance = torch.norm(preferences[users] - pos_item_emb, p=2, dim=1)

        # Negative distance (how far the user preference is from each negative item embedding)
        negative_distance = torch.norm(preferences[users].unsqueeze(1) - neg_emb, p=2, dim=2)

        # Compute the triplet loss (maximize the positive distance, minimize the negative distance)
        loss = F.relu(positive_distance - negative_distance + self.margin)  # hinge loss
        
        loss = loss / self.temperature
        
        # Mean loss over the batch
        triplet_loss = loss.mean()  # Mean loss across batch
        
        return triplet_loss


class Our_GCNs(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Our_GCNs, self).__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, weight_vector, size=None):
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out


class Nonlinear_GCNs(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Nonlinear_GCNs, self).__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, weight_vector, size=None):
        x = torch.matmul(x, self.weight)
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out


class CrossAttentionLayer(nn.Module):
    def __init__(self, combined_embedding_dim, num_heads, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.attention_image = nn.MultiheadAttention(embed_dim=combined_embedding_dim // 2, num_heads=num_heads, dropout=dropout)
        self.attention_text = nn.MultiheadAttention(embed_dim=combined_embedding_dim // 2, num_heads=num_heads, dropout=dropout)
        self.modal_weights = nn.Parameter(torch.Tensor([0.5, 0.5]))  # Learnable modality weight
        self.softmax = nn.Softmax(dim=0)

    def forward(self, user_embedding, image_embedding, text_embedding):
        device = user_embedding.device
        image_embedding = image_embedding.to(device)
        text_embedding = text_embedding.to(device)

        # Reshape for Multihead Attention
        user_embedding = user_embedding.unsqueeze(0).permute(1, 0, 2)
        image_embedding = image_embedding.unsqueeze(0).permute(1, 0, 2)
        text_embedding = text_embedding.unsqueeze(0).permute(1, 0, 2)

        # Compute attention separately for image and text
        image_attention_output, image_weights = self.attention_image(
            query=user_embedding, key=image_embedding, value=image_embedding
        )
        text_attention_output, text_weights = self.attention_text(
            query=user_embedding, key=text_embedding, value=text_embedding
        )

        # Remove sequence dimension
        image_attention_output = image_attention_output.squeeze(1)
        text_attention_output = text_attention_output.squeeze(1)

        # Concatenate outputs instead of weighted sum
        fused_attention_output = torch.cat([image_attention_output, text_attention_output], dim=-1)  # (batch_size, 512)

        return fused_attention_output, image_weights, text_weights


class MeGCN(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        n_layers,
        has_norm,
        feat_embed_dim,
        nonzero_idx,
        image_feats,
        text_feats,
        alpha,
        agg,
        cf,
        cf_gcn,
        lightgcn,
        margin,
        temperature,
        use_contrastive,
        use_cross_attention,
        use_mlp,
        user_item_interactions=None, 
    ):
        super(MeGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.has_norm = has_norm
        self.feat_embed_dim = feat_embed_dim
        self.nonzero_idx = torch.tensor(nonzero_idx).cuda().long().T
        self.alpha = alpha
        self.agg = agg
        self.cf = cf
        self.cf_gcn = cf_gcn
        self.lightgcn = lightgcn
        self.margin = margin
        self.temperature = temperature
        self.use_contrastive= use_contrastive
        self.use_cross_attention= use_cross_attention
        self.use_mlp= use_mlp

        # Added user-item interaction data for visualization
        self.user_item_interactions = user_item_interactions if user_item_interactions is not None else {}
        
        # Triplet Loss with a given temperature
        self.triplet_loss= TripletLoss(margin=margin, temperature=temperature)

        self.image_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
        self.text_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
        nn.init.xavier_uniform_(self.image_preference.weight)
        nn.init.xavier_uniform_(self.text_preference.weight)

        self.image_embedding = nn.Embedding.from_pretrained(
            torch.tensor(image_feats, dtype=torch.float), freeze=True
        )  # [# of items, 4096]
        self.text_embedding = nn.Embedding.from_pretrained(
            torch.tensor(text_feats, dtype=torch.float), freeze=True
        )  # [# of items, 1024]

        if self.cf:
            self.user_embedding = nn.Embedding(self.n_users, self.feat_embed_dim)
            self.item_embedding = nn.Embedding(self.n_items, self.feat_embed_dim)
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)

        self.image_trs = nn.Linear(image_feats.shape[1], self.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], self.feat_embed_dim)

        if not self.cf:
            if self.agg == "fc":
                self.transform = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
            elif self.agg == "weighted_sum":
                self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
                self.softmax = nn.Softmax(dim=0)
        else:
            if self.agg == "fc":
                self.transform = nn.Linear(self.feat_embed_dim * 3, self.feat_embed_dim)
            elif self.agg == "weighted_sum":
                self.modal_weight = nn.Parameter(torch.Tensor([0.33, 0.33, 0.33]))
                self.softmax = nn.Softmax(dim=0)

        self.layers = nn.ModuleList(
            [
                Our_GCNs(self.feat_embed_dim, self.feat_embed_dim)
                for _ in range(self.n_layers)
            ]
        )

        # Cross-Attention Layer
        combined_embedding_dim = feat_embed_dim * 2  # Combined size of image and text embeddings
        self.cross_attention = CrossAttentionLayer(combined_embedding_dim=combined_embedding_dim, num_heads=1)

        # Add a transformation layer to align output with MLP input
        self.cross_attention_transform = nn.Linear(combined_embedding_dim, 128)  # Map to 128 features

        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),  # Increased hidden size
            nn.ReLU(),
            nn.BatchNorm1d(256, track_running_stats=False),
            nn.Dropout(0.4),

            nn.Linear(256, 128),  # Another hidden layer
            nn.ReLU(),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.Dropout(0.4),

            nn.Linear(128, 1)      # Final output layer
        )


    def compute_mmd(self, source, target, sigma=1.0):
        def pairwise_distances(x, y):
            x_norm = (x ** 2).sum(dim=1).unsqueeze(1)  # [B, 1]
            y_norm = (y ** 2).sum(dim=1).unsqueeze(0)  # [1, B]
            dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.t())
            return torch.clamp(dist, 0.0, float('inf'))  # Numerical stability

        def gaussian_kernel_matrix(x, y, sigma):
            pairwise_dists = pairwise_distances(x, y)
            return torch.exp(-pairwise_dists / (2 * sigma ** 2))

        Kxx = gaussian_kernel_matrix(source, source, sigma)
        Kyy = gaussian_kernel_matrix(target, target, sigma)
        Kxy = gaussian_kernel_matrix(source, target, sigma)

        # Remove diagonals from Kxx and Kyy if they cause bias (optional)
        B = source.size(0)
        Kxx = (Kxx.sum() - Kxx.diag().sum()) / (B * (B - 1))
        Kyy = (Kyy.sum() - Kyy.diag().sum()) / (B * (B - 1))
        Kxy = Kxy.mean()

        return Kxx + Kyy - 2 * Kxy


    def forward(self, edge_index, edge_weight, _eval=False, pos_items=None, neg_items=None, users=None, save_debug=False):
        
        # transform
        image_emb = self.image_trs(self.image_embedding.weight)  # [# of items, feat_embed_dim]
        text_emb = self.text_trs(self.text_embedding.weight)  # [# of items, feat_embed_dim]

        # Extract positive and negative item embeddings for each modality
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=image_emb.device) if isinstance(pos_items, list) else pos_items
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=image_emb.device) if isinstance(neg_items, list) else neg_items

        # Extract positive and negative item embeddings for each modality
        pos_item_emb_image = image_emb[pos_items]  # Shape: [batch_size, embed_dim]
        neg_item_emb_image = image_emb[neg_items]  # Shape: [batch_size, num_negatives, embed_dim]

        pos_item_emb_text = text_emb[pos_items]  # Shape: [batch_size, embed_dim]
        neg_item_emb_text = text_emb[neg_items]  # Shape: [batch_size, num_negatives, embed_dim]

        if self.cf:
            # Collaborative filtering item embeddings
            pos_item_emb_cf = self.item_embedding.weight[pos_items]  # Shape: [batch_size, embed_dim]
            neg_item_emb_cf = self.item_embedding.weight[neg_items]  # Shape: [batch_size, num_negatives, embed_dim]
        
        if self.has_norm:
            image_emb = F.normalize(image_emb)
            text_emb = F.normalize(text_emb)

        image_preference = self.image_preference.weight
        text_preference = self.text_preference.weight

        # propagate
        ego_image_emb = torch.cat([image_preference, image_emb], dim=0)
        ego_text_emb = torch.cat([text_preference, text_emb], dim=0)

        # Collaborative filtering embeddings
        if self.cf:
            user_emb = self.user_embedding.weight
            item_emb = self.item_embedding.weight
            ego_cf_emb = torch.cat([user_emb, item_emb], dim=0)
            if self.cf_gcn == "LightGCN":
                all_cf_emb = [ego_cf_emb]

        if self.lightgcn:
            all_image_emb = [ego_image_emb]
            all_text_emb = [ego_text_emb]

        # Propagate through GCN layers
        for layer in self.layers:
            if not self.lightgcn:
                side_image_emb = layer(ego_image_emb, edge_index, edge_weight)
                side_text_emb = layer(ego_text_emb, edge_index, edge_weight)

                ego_image_emb = side_image_emb + self.alpha * ego_image_emb
                ego_text_emb = side_text_emb + self.alpha * ego_text_emb
            else:
                side_image_emb = layer(ego_image_emb, edge_index, edge_weight)
                side_text_emb = layer(ego_text_emb, edge_index, edge_weight)
                ego_image_emb = side_image_emb
                ego_text_emb = side_text_emb
                all_image_emb += [ego_image_emb]
                all_text_emb += [ego_text_emb]
            if self.cf:
                if self.cf_gcn == "MeGCN":
                    side_cf_emb = layer(ego_cf_emb, edge_index, edge_weight)
                    ego_cf_emb = side_cf_emb + self.alpha * ego_cf_emb
                elif self.cf_gcn == "LightGCN":
                    side_cf_emb = layer(ego_cf_emb, edge_index, edge_weight)
                    ego_cf_emb = side_cf_emb
                    all_cf_emb += [ego_cf_emb]

        # Final embedding splits for LightGCN or MeGCN
        if not self.lightgcn:
            final_image_preference, final_image_emb = torch.split(
                ego_image_emb, [self.n_users, self.n_items], dim=0
            )
            final_text_preference, final_text_emb = torch.split(
                ego_text_emb, [self.n_users, self.n_items], dim=0
            )
        else:
            all_image_emb = torch.stack(all_image_emb, dim=1)
            all_image_emb = all_image_emb.mean(dim=1, keepdim=False)
            final_image_preference, final_image_emb = torch.split(
                all_image_emb, [self.n_users, self.n_items], dim=0
            )

            all_text_emb = torch.stack(all_text_emb, dim=1)
            all_text_emb = all_text_emb.mean(dim=1, keepdim=False)
            final_text_preference, final_text_emb = torch.split(
                all_text_emb, [self.n_users, self.n_items], dim=0
            )

        if self.cf:
            if self.cf_gcn == "MeGCN":
                final_cf_user_emb, final_cf_item_emb = torch.split(
                    ego_cf_emb, [self.n_users, self.n_items], dim=0
                )
            elif self.cf_gcn == "LightGCN":
                all_cf_emb = torch.stack(all_cf_emb, dim=1)
                all_cf_emb = all_cf_emb.mean(dim=1, keepdim=False)
                final_cf_user_emb, final_cf_item_emb = torch.split(
                    all_cf_emb, [self.n_users, self.n_items], dim=0
                )

        # Return early for evaluation
        if _eval:
            return ego_image_emb, ego_text_emb

        # Aggregation options
        if not self.cf:
            if self.agg == "concat":
                items = torch.cat(
                    [final_image_emb, final_text_emb], dim=1
                )  # [# of items, feat_embed_dim * 2] [Concatenate modality embeddings]
                user_preference = torch.cat(
                    [final_image_preference, final_text_preference], dim=1
                )  # [# of users, feat_embed_dim * 2]
            elif self.agg == "sum":
                items = final_image_emb + final_text_emb  # [# of items, feat_embed_dim] [Sum modality embeddings]
                user_preference = (
                    final_image_preference + final_text_preference
                )  # [# of users, feat_embed_dim]
            elif self.agg == "weighted_sum":
                weight = self.softmax(self.modal_weight)
                items = (
                    weight[0] * final_image_emb + weight[1] * final_text_emb
                )  # [# of items, feat_embed_dim]  [Weighted sum]
                user_preference = (
                    weight[0] * final_image_preference
                    + weight[1] * final_text_preference
                )  # [# of users, feat_embed_dim]
            elif self.agg == "fc":
                items = self.transform(
                    torch.cat([final_image_emb, final_text_emb], dim=1)
                )  # [# of items, feat_embed_dim] Project concatenated embedding
                user_preference = self.transform(
                    torch.cat([final_image_preference, final_text_preference], dim=1)
                )  # [# of users, feat_embed_dim]
        else:
            # Includes collaborative filtering embeddings
            if self.agg == "concat":
                items = torch.cat(
                    [final_image_emb, final_text_emb, final_cf_item_emb], dim=1
                )  # [# of items, feat_embed_dim * 2]
                user_preference = torch.cat(
                    [final_image_preference, final_text_preference, final_cf_user_emb],
                    dim=1,
                )  # [# of users, feat_embed_dim * 2]
            elif self.agg == "sum":
                items = (
                    final_image_emb + final_text_emb + final_cf_item_emb
                )  # [# of items, feat_embed_dim]
                user_preference = (
                    final_image_preference + final_text_preference + final_cf_user_emb
                )  # [# of users, feat_embed_dim]
            elif self.agg == "weighted_sum":
                weight = self.softmax(self.modal_weight)
                items = (
                    weight[0] * final_image_emb
                    + weight[1] * final_text_emb
                    + weight[2] * final_cf_item_emb
                )  # [# of items, feat_embed_dim]
                user_preference = (
                    weight[0] * final_image_preference
                    + weight[1] * final_text_preference
                    + weight[2] * final_cf_user_emb
                )  # [# of users, feat_embed_dim]
            elif self.agg == "fc":
                items = self.transform(
                    torch.cat(
                        [final_image_emb, final_text_emb, final_cf_item_emb], dim=1
                    )
                )  # [# of items, feat_embed_dim]
                user_preference = self.transform(
                    torch.cat(
                        [
                            final_image_preference,
                            final_text_preference,
                            final_cf_user_emb,
                        ],
                        dim=1,
                    )
                )  # [# of users, feat_embed_dim]

        # Compute contrastive loss (only during training)
        contrastive_loss = None
        if not _eval and pos_items is not None and neg_items is not None:
            if self.use_contrastive:
                image_contrastive_loss = self.triplet_loss(
                    pos_items, users, final_image_preference, pos_item_emb_image, neg_item_emb_image
                )
                text_contrastive_loss = self.triplet_loss(
                    pos_items, users, final_text_preference, pos_item_emb_text, neg_item_emb_text
                )
                if self.cf:
                    cf_contrastive_loss = self.triplet_loss(
                        pos_items, users, final_cf_user_emb, pos_item_emb_cf, neg_item_emb_cf
                    )
                    contrastive_loss = image_contrastive_loss + text_contrastive_loss + cf_contrastive_loss
                else:
                    contrastive_loss = image_contrastive_loss + text_contrastive_loss

        # ======== MMD DOMAIN ADAPTATION: align text and image distributions ========
        mmd_loss = None
        if not _eval:
            mmd_loss = self.compute_mmd(final_text_emb, final_image_emb)

        # Cross-attention mechanism
        if self.use_cross_attention:
            attention_output, image_weights, text_weights = self.cross_attention(
                final_image_preference, final_image_emb, final_text_emb
            )
            attention_output = self.cross_attention_transform(attention_output)  # (batch_size, 128)
        else:
            attention_output = torch.cat((final_image_emb, final_text_emb), dim=-1)  # (batch_size, 512)
            attention_output = self.cross_attention_transform(attention_output)  # Reduce to (batch_size, 128)
            image_weights, text_weights = None, None

        # MLP for final prediction
        if self.use_mlp:
            prediction = self.mlp(attention_output).squeeze(-1)
        else:
            prediction = torch.sum(attention_output, dim=-1)  # Summation-based prediction if MLP is disabled

        if save_debug:
            os.makedirs("debug_data", exist_ok=True)
            np.save("debug_data/embeddings.npy", user_preference.cpu().detach().numpy())
            np.save("debug_data/attention_weights.npy", attention_output.cpu().detach().numpy())
            np.save("debug_data/mlp_inputs.npy", attention_output.cpu().detach().numpy())
            print("Saved embeddings, attention weights, and MLP inputs for visualization.")

        return user_preference, items, prediction, contrastive_loss, image_weights, text_weights, mmd_loss



class MONET(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        feat_embed_dim,
        nonzero_idx,
        has_norm,
        image_feats,
        text_feats,
        n_layers,
        alpha,
        beta,
        agg,
        cf,
        cf_gcn,
        lightgcn,
        margin,
        temperature,
        lambda_mmd,
        use_contrastive,
        use_cross_attention,
        use_mlp,
    ):
        super(MONET, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.feat_embed_dim = feat_embed_dim
        self.n_layers = n_layers
        self.nonzero_idx = nonzero_idx
        self.alpha = alpha
        self.beta = beta
        self.agg = agg
        self.image_feats = torch.tensor(image_feats, dtype=torch.float).cuda()
        self.text_feats = torch.tensor(text_feats, dtype=torch.float).cuda()
        self.cf = cf
        self.cf_gcn = cf_gcn
        self.lightgcn = lightgcn
        self.margin = margin
        self.has_norm = has_norm
        self.temperature = temperature
        self.lambda_mmd=lambda_mmd
        self.use_contrastive=use_contrastive
        self.use_cross_attention=use_cross_attention
        self.use_mlp=use_mlp

        self.megcn = MeGCN(
            self.n_users,
            self.n_items,
            self.n_layers,
            has_norm,
            self.feat_embed_dim,
            self.nonzero_idx,
            image_feats,
            text_feats,
            self.alpha,
            self.agg,
            cf,
            cf_gcn,
            lightgcn,
            temperature,
            lambda_mmd,
            use_contrastive,
            use_cross_attention,
            use_mlp,
        )

        nonzero_idx = torch.tensor(self.nonzero_idx).cuda().long().T
        nonzero_idx[1] = nonzero_idx[1] + self.n_users
        self.edge_index = torch.cat(
            [nonzero_idx, torch.stack([nonzero_idx[1], nonzero_idx[0]], dim=0)], dim=1
        )
        self.edge_weight = torch.ones((self.edge_index.size(1))).cuda().view(-1, 1)
        self.edge_weight = normalize_laplacian(self.edge_index, self.edge_weight)

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

    def forward(self, _eval=False, pos_items=None, neg_items=None, users=None, save_debug=False):
        if _eval:
            img, txt = self.megcn(self.edge_index, self.edge_weight, _eval=True)
            return {"image": img, "text": txt}

        user_preference, items, prediction, contrastive_loss, image_weights, text_weights, mmd_loss  = self.megcn(self.edge_index, self.edge_weight, _eval=False, pos_items=pos_items, neg_items=neg_items, users=users, save_debug=save_debug
        )
        
        return user_preference, items, prediction, contrastive_loss, image_weights, text_weights, mmd_loss

    def bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items, target_aware):
        current_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]

        if target_aware:
            # target-aware loss calculation
            item_item = torch.mm(item_emb, item_emb.T)
            pos_item_query = item_item[pos_items, :]  # (batch_size, n_items)
            neg_item_query = item_item[neg_items, :]  # (batch_size, n_items)
            pos_target_user_alpha = F.relu(
                torch.multiply(pos_item_query, self.adj[users, :]).masked_fill(
                    self.adj[users, :] == 0, -1e9
                ),
            )  # (batch_size, n_items)
            neg_target_user_alpha = F.relu(
                torch.multiply(neg_item_query, self.adj[users, :]).masked_fill(
                    self.adj[users, :] == 0, -1e9
                ),
            )  # (batch_size, n_items)
            pos_target_user = torch.mm(
                pos_target_user_alpha, item_emb
            )  # (batch_size, dim)
            neg_target_user = torch.mm(
                neg_target_user_alpha, item_emb
            )  # (batch_size, dim)

            # predictor
            pos_scores = (1 - self.beta) * torch.sum(
                torch.mul(current_user_emb, pos_item_emb), dim=1
            ) + self.beta * torch.sum(torch.mul(pos_target_user, pos_item_emb), dim=1)
            neg_scores = (1 - self.beta) * torch.sum(
                torch.mul(current_user_emb, neg_item_emb), dim=1
            ) + self.beta * torch.sum(torch.mul(neg_target_user, neg_item_emb), dim=1)
        else:
            pos_scores = torch.sum(torch.mul(current_user_emb, pos_item_emb), dim=1)
            neg_scores = torch.sum(torch.mul(current_user_emb, neg_item_emb), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        regularizer = (
            0.5 * (pos_item_emb**2).sum()
            + 0.5 * (neg_item_emb**2).sum()
            + 0.5 * (current_user_emb**2).sum()
        )
        emb_loss = regularizer / pos_item_emb.size(0)

        reg_loss = 0.0

        return mf_loss, emb_loss, reg_loss