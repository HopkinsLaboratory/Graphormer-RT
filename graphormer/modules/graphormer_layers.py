# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long).to('cuda:0')
    print(feature_offset)
    x = x + feature_offset
    return x

class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms

        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms+1, hidden_dim, padding_idx=0)

        self.num_features = 67 # Tanaka _ hsmb = 78 ## need to set this manually right now, will include in command line in future updates

        sizes = [45, hidden_dim // 2, hidden_dim] ## number of layers and size of projections for learned node embeddings

        p = 0.15 ## learned feature encoder dropout rate
        size_int = [59,   hidden_dim // 2, hidden_dim] ## size of projection for learned node embeddings
        size_float = [30,  hidden_dim // 2, hidden_dim] ## size of projection for learned node embeddings
        ## 11 without s1, s2

        graph_sizes = [45, hidden_dim // 2, hidden_dim] ## size of projection for graph encoder
        physical_sizes = [39, hidden_dim //2, hidden_dim]## size of projection for graph encoder
        p_physico = 0.15
        self.float_encoder = nn.ModuleList([nn.Sequential(nn.Linear(x, y),nn.Dropout(p = p)) for x, y in zip(sizes[0:-1], sizes[1:])]) ## Build Graph Feature Encoder
        # self.graph_encoder = nn.ModuleList([nn.Sequential(nn.Linear(x, y),nn.Dropout(p = p_graph)) for x, y in zip(graph_sizes[0:-1], graph_sizes[1:])]) ## Build Graph Encoder
        self.column_float_encoder = nn.ModuleList([nn.Sequential(nn.Linear(x, y),nn.Dropout(p = p)) for x, y in zip(size_float[0:-1], size_float[1:])]) ## Build Column Feature Encoder
        self.column_int_encoder = nn.ModuleList([nn.Sequential(nn.Linear(x, y),nn.Dropout(p = p)) for x, y in zip(size_int[0:-1], size_int[1:])]) ## Build Column Graph Encoder

        self.physico_encoder = nn.ModuleList([nn.Sequential(nn.Linear(x, y),nn.Dropout(p = p_physico)) for x, y in zip(physical_sizes[0:-1], physical_sizes[1:])]) ## Build Column Graph Encoder



        # for seq in self.physico_encoder.children():
        #     nn.init.xavier_normal_(seq[0].weight)


        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):

        x, in_degree, out_degree = (
            batched_data["x"], 
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]
        int_feature = torch.unsqueeze(x[:,:, 0], dim = 2).long() ## used in cases where testing atom number, other embeddings


        float_feature = x[:,:, :].unsqueeze(2) ## grabbing all node features


        ## Grabbing column node
        condition = (float_feature[:, :, 0, 0] == -1) ## we label the global node to start with -1 so we can find it here
        mask = condition.unsqueeze(-1).unsqueeze(-1).expand(*list(float_feature.size())) ## locate all the global nodes
        global_feat= float_feature * mask
        global_feat[global_feat == -1] = 0 ## getting rid of the negative values, was only a label \

        ## Grabbing global physicochemical properties from secondary global node

        # condition_2 = (float_feature[:, :, 0, 0] == -2) ## we label the global node to start with -1 so we can find it here
        # mask2 = condition_2.unsqueeze(-1).unsqueeze(-1).expand(*list(float_feature.size())) ## locate all the global nodes
        # mol_feat = float_feature * mask2
        
        # print(mol_feat[0, -1, :, :])
        # mol_feat[mol_feat < 0] = 0 ## getting rid of the negative values, was only a label 
        
        # global_feat[mask2] = 0
        # float_feature[mask2] = 0 ## getting rid of the global node "values" from float_feature


        float_feature[mask] = 0 ## getting rid of the global node "values" from float_feature
        ## 44 is atomic mass
        ## 43 is partial charge
        ## 37 - 42 is total_bonds
        ## 29 - 36 is explicit valence
        ## 24 - 29 is number of H
        ## 23 is aromaticity
        ## 17 - 23 is hybridization
        ## 11 - 17 is formal charge
        ## 0 - 11 is atomic number
        float_feature = float_feature[:, :, :, :45] ## removing the padded zeroes ## atomic float features
        # print(float_feature[0, :, :, 44])
        float_feature[:, :, :, 44] = 0 ## removing the atomic mass
        # float_feature[:, :, :, 43] = 0 ## removing the partial charge
        # float_feature[:, :, :, 37:43] = 0 ## removing the total bonds
        # float_feature[:, :, :, 29:37] = 0 ## removing the explicit valence YES
        # float_feature[:, :, :, 24:29] = 0 ## removing the number of H
        # float_feature[:, :, :, 23] = 0 ## removing the aromaticity
        # float_feature[:, :, :, 17:23] = 0 ## removing the hybridization YES
        float_feature[:, :, :, 11:17] = 0 ## removing the formal charge YES
        # float_feature[:, :, :, 0:11] = 0 ## removing the atomic cnumber


        # print(float_feature[0, :, :, 44])
        # exit()

        # float_feature = float_feature[:, :, :, 1:] ## removing the atomic number
        
        global_int_feat = global_feat[:, :, :,1:60] ## 56 is the bound for the one hot encoded features
        global_float_feat = global_feat[:, :, :, 60:90]  ## 67 originally ## TODO: Figure out why tanaka and hsmb features disappear when they appear here
        # global_physico_feat = global_feat[:, :, :, 67:]
        
        # print(global_float_feat[0, -1, :, :])

        # global_int_feat[:,:,:,40:] = 0 ## Additives B
        
        # print(global_float_feat[0, -1, :, :])
        global_float_feat[:,:,:,15:] = 0 ## removing the tanaka and hsmb features 11 ## 12 does no good. 


        global_float_feat[:,:,:,10] = 0 # pHB ## add back in for RP predictions
        global_float_feat[:,:,:,8] = 0 # B3
        # global_float_feat[:,:,:,12] = 0 # temperature

        # global_float_feat[:,:,:,13] = 0 # ABLATE FLOW RATE HERE

        # print(global_float_feat[0, -1, :, :])
        # exit()
        # print(global_float_feat[0, -1, :, :])
        # print(global_float_feat[0, -1, :, 11])
        # # exit()
        # global_float_feat[:,:,:,4] = 0 

        ### THIS RESTORES THE TANAKA AND HSMB FEATURE ABLATIONS
        # global_float_feat[:,:,:,12:16] = 0 ## removing the tanaka and hsmb features 11 ## 12 does no good. 
        # global_float_feat[:,:,:,18:] = 0 ## removing the tanaka and hsmb features 11
        
        # print(global_float_feat[0, -1, :, :].shape)
        # print(global_float_feat[0, -1, :, :])
        # exit()

        # print(global_float_feat[0, -1, :, :])
        # print(len(global_float_feat[0, -1, :, :]))
        # exit()

        for y in self.float_encoder: ## applying  learned node embedding encoder
            float_feature = y(float_feature)
            float_feature = F.relu(float_feature) ## TODO: replace with GElu

        for g in self.column_int_encoder: ## seperate encoder for global node
            global_int_feat = g(global_int_feat)
            global_int_feat = F.relu(global_int_feat)

        for g in self.column_float_encoder: ## seperate encoder for global node
            global_float_feat = g(global_float_feat)
            global_float_feat = F.relu(global_float_feat)

        # for r in self.physico_encoder: ## seperate encoder for global node
        #     global_physico_feat = r(global_physico_feat)
        #     global_physico_feat= F.relu(global_physico_feat)


        global_int_feat = global_int_feat.squeeze(2) ## reducing dimensions
        global_float_feat = global_float_feat.squeeze(2) ## reducing dimensions
        float_feature = float_feature.squeeze(2) ## reducing dimensions
        # global_physico_feat = global_physico_feat.squeeze(2) ## reducing dimensions

        node_feature = (
            # int_feature +
            float_feature 
            # + global_physico_feat 
            + global_int_feat
            + global_float_feat
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)

        )
        del float_feature, global_feat, mask,  condition, int_feature # reclaiming memory (mask might be creating a view that is not being deleted)
        # del mask2, condition_2, mol_feat, global_int_feat, global_float_feat
        # if epoch % 10 == 0:
        # gc.collect()

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature

class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias