import torch
import torch.nn.functional as F
import torch_geometric
from pathlib import Path
import gzip
import pickle
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix
from config import *
import math
import torch.nn as nn
DEVICE = Config.DEVICE
# DEVICE = torch.device("cuda:1")

def create_A_np(row, col, data, size_):
    num_row = size_[0]
    num_col = size_[1]
    row_m1, col_m1, data_m1 = np.arange(num_col), np.arange(num_col), np.ones(num_col)
    row_m2, col_m2 = row + num_col, col
    row_m3, col_m3 = col, row + num_col
    row_m4, col_m4, data_m4 = np.arange(num_row)+num_col, np.arange(num_row)+num_col, np.ones(num_row)
    row_A = np.concatenate((row_m1, row_m2, row_m3, row_m4))
    col_A = np.concatenate((col_m1, col_m2, col_m3, col_m4))
    data_A = np.concatenate((data_m1, data, data, data_m4))
    return coo_matrix((data_A, (row_A, col_A)), shape=(num_row+num_col, num_row+num_col)).toarray()

def create_A_tensor(row, col, data, size_):
    num_row = size_[0]
    num_col = size_[1]
    row_m1, col_m1, data_m1 = torch.arange(num_col), torch.arange(num_col), torch.ones(num_col)
    row_m2, col_m2 = row + num_col, col
    row_m3, col_m3 = col, row + num_col
    row_m4, col_m4, data_m4 = torch.arange(num_row)+num_col, torch.arange(num_row)+num_col, torch.ones(num_row)
    row_A = torch.cat((row_m1.to(Config.DEVICE), row_m2, row_m3, row_m4.to(Config.DEVICE)))
    col_A = torch.cat((col_m1.to(Config.DEVICE), col_m2, col_m3, col_m4.to(Config.DEVICE)))
    data_A = torch.cat((data_m1.to(Config.DEVICE), data, data, data_m4.to(Config.DEVICE)))
    return torch.sparse_coo_tensor(torch.vstack((row_A, col_A)), data_A,
                                   (num_row+num_col, num_row+num_col)).to_dense()

# def create_A_tensor_2(row, col, data, size_):
#     num_row = size_[0]
#     num_col = size_[1]
#     row_m1, col_m1, data_m1 = torch.arange(num_col), torch.arange(num_col), torch.ones(num_col)
#     row_m2, col_m2 = row + num_col, col
#     row_m3, col_m3 = col, row + num_col
#     row_m4, col_m4, data_m4 = torch.arange(num_row)+num_col, torch.arange(num_row)+num_col, torch.ones(num_row)
#     row_A = torch.cat((row_m2, row_m3))
#     col_A = torch.cat((col_m2, col_m3))
#     data_A = torch.cat((data, data))
#     return torch.vstack((row_A, col_A)), data_A

def transfer_edge(row, col, data, len_col):
    e1 = torch.cat((row + len_col, col))
    e2 = torch.cat((col, row + len_col))
    return torch.vstack((e1, e2)), torch.cat((data, data))


class PreNormException(Exception):
    pass

class PreNormLayer(torch.nn.Module):
    """
    Our pre-normalization layer, whose purpose is to normalize an input layer
    to zero mean and unit variance to speed-up and stabilize GCN training. The
    layer's parameters are aimed to be computed during the pre-training phase.
    """
    def __init__(self, n_units, shift=True, scale=True):
        super(PreNormLayer, self).__init__()
        assert shift or scale

        if shift:
            self.register_buffer(f"shift", torch.zeros((n_units,), dtype=torch.float32))
        else:
            self.shift = None

        if scale:
            self.register_buffer(f"scale", torch.ones((n_units,), dtype=torch.float32))
        else:
            self.scale = None

        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input):
        if self.waiting_updates:
            self.update_stats(input)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input = input + self.shift

        if self.scale is not None:
            input = input * self.scale

        return input

    def start_updates(self):
        """
        Initializes the pre-training phase.
        """
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input.shape[-1]}."

        input = input.reshape([-1, self.n_units])
        sample_avg = torch.mean(input, dim=0)
        sample_var = torch.mean((input - sample_avg) ** 2, dim=0)
        sample_count = input.numel() / self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """

        assert self.count > 0
        if self.shift is not None:
            self.shift = - self.avg

        if self.scale is not None:
            self.var = torch.where(torch.eq(self.var, 0.0), torch.ones_like(self.var), self.var) # NaN check trick
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False

class BaseModel(torch.nn.Module):
    def initialize_parameters(self):
        for l in self.modules():
            if isinstance(l, torch.nn.Linear):
                self.initializer(l.weight.data)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias.data, 0)

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train(self, state):
        with torch.no_grad():
            try:
                self.forward(*state)
                return False
            except PreNormException:
                return True

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) \
                    and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def save_state(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore_state(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))



from torch_geometric.nn import GraphConv, GCNConv, GCN2Conv, ARMAConv, APPNP, GraphUNet, TAGConv

class GNNPolicyMatrix(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19
        gcn_layer_num = 3

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),

        )

        self.edge_feautre = torch.nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1, bias=True),
            nn.LayerNorm(1)
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )


        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size*2, int(emb_size)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(emb_size), 1, bias=False),
        )

        self.gnn1 = GraphConv(emb_size, emb_size, "mean")
        self.l1 = nn.Linear(emb_size*2, emb_size)
        self.gnn2 = GraphConv(emb_size, emb_size, "mean")
        self.l2 = nn.Linear(emb_size*2, emb_size)
        self.gnn3 = GraphConv(emb_size, emb_size, "mean")
        self.l3 = nn.Linear(emb_size*2, emb_size)
        self.gnn4 = GraphConv(emb_size, emb_size, "mean")
        self.l4 = nn.Linear(emb_size*2, emb_size)


        self.layer_norm = nn.LayerNorm(emb_size)


    def forward(self, constraint_features, edge_indices_0, edge_indices_1, edge_features, variable_features,
                tree_features=None, candidates=None, nb_candidates=None):

        # edge_indices = torch.stack([edge_indices_0, edge_indices_1], dim=0)
        edge_norm = self.edge_embedding(edge_features).squeeze(-1)

        # A = create_A_tensor(edge_indices_0, edge_indices_1, edge_features,
        #                     (constraint_features.size(0), variable_features.size(0)))
        edge_i, edge_w = transfer_edge(edge_indices_0, edge_indices_1, edge_norm, variable_features.size(0))

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        variable_features = self.var_embedding(variable_features)
        x = torch.vstack((variable_features, constraint_features))
        # node_embed = self.gcn_forward(node_embed, A)
        # probs = self.output_module(node_embed[:variable_features.size(0)]).squeeze(-1)

        edge_features = self.edge_feautre(edge_w.unsqueeze(-1)).squeeze(-1) + edge_w

        x_out = self.gcn_forward(x, edge_i, edge_features)

        x_out = torch.cat((x_out, x), -1)
        probs = self.output_module(x_out[:variable_features.size(0)])


        return probs.squeeze(-1)

    def gcn_forward(self, x, edge_i, w):
        x_out1 = self.layer_norm(self.gnn1(self.l1(torch.cat((x, x), -1)), edge_i, w))
        x_out2 = self.layer_norm(self.gnn2(self.l2(torch.cat((x, x_out1), -1)), edge_i, w))
        x_out3 = self.layer_norm(self.gnn3(self.l3(torch.cat((x_out1, x_out2), -1)), edge_i, w))
        x_out4 = self.layer_norm(self.gnn4(self.l4(torch.cat((x_out2, x_out3), -1)), edge_i, w))
        return x_out4
        # node_embed_ini = node_embed
        # node_embed = node_embed_ini
        # for layer in self.gcn_layers:
        #     node_embed_ = torch.mm(A, layer(node_embed))
        #     node_embed = node_embed + node_embed_
        #     node_embed = self.norm_layer(node_embed)
        # node_embed = node_embed + node_embed_ini
        # node_embed = self.norm_layer(node_embed)
        # return node_embed


class GNNPolicy2(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),

        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # PSEcost EMBEDDING
        self.pse_embedding = torch.nn.Sequential(
            PreNormLayer(1),
            torch.nn.Linear(1, emb_size),
            torch.nn.ReLU()
        )

        # self.gcn_layers = torch.nn.ModuleList([torch.nn.Linear(emb_size, emb_size, bias=False)
        #                                        for i in range(gcn_layer_num)])
        # self.norm_layer = torch.nn.LayerNorm(emb_size)

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size*2, int(emb_size)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(emb_size), 1, bias=False),
        )

        self.gnn1 = GraphConv(emb_size, emb_size, "mean")
        self.l1 = nn.Linear(emb_size*2, emb_size)
        self.gnn2 = GraphConv(emb_size, emb_size, "mean")
        self.l2 = nn.Linear(emb_size*2, emb_size)
        self.gnn3 = GraphConv(emb_size, emb_size, "mean")
        self.l3 = nn.Linear(emb_size*2, emb_size)
        # self.gnn4 = GraphConv(emb_size, emb_size, "mean")
        # self.l4 = nn.Linear(emb_size*2, emb_size)

        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, constraint_features, edge_indices_0, edge_indices_1, edge_features, variable_features,
                tree_features=None, candidates=None, nb_candidates=None, nb_vars=None):

        # variable_features = torch.cat((variable_features, tree_features[3].unsqueeze(-1)), -1)
        # embed
        constraint_features = self.cons_embedding(constraint_features)
        variable_features = self.var_embedding(variable_features)
        edge_features = self.edge_embedding(edge_features).squeeze(-1)

        edge_i, edge_w = transfer_edge(edge_indices_0, edge_indices_1, edge_features, variable_features.size(0))
        x = torch.vstack((variable_features, constraint_features))

        # First step: linear embedding layers to a common dimension (64)

        x_out = self.gcn_forward(x, edge_i, edge_w)

        pse = self.pse_embedding(tree_features[3].unsqueeze(-1))
        x_out = torch.cat((x_out[candidates], x[candidates] + pse), -1)
        probs = self.output_module(x_out)

        return probs.squeeze(-1)

    def gcn_forward(self, x, edge_i, w):
        x_out1 = self.layer_norm(self.gnn1(self.l1(torch.cat((x, x), -1)), edge_i, w))
        x_out2 = self.layer_norm(self.gnn2(self.l2(torch.cat((x, x_out1), -1)), edge_i, w))
        x_out3 = self.layer_norm(self.gnn3(self.l3(torch.cat((x_out1, x_out2), -1)), edge_i, w))
        return x_out3



class GNNPointerPolicy(GNNPolicyMatrix):
    def __init__(self):
        super().__init__()
        emb_size = 128
        tree_nfeats = 10

        self.key_project = torch.nn.Linear(emb_size, emb_size)
        self.value_project = torch.nn.Linear(emb_size, emb_size)
        self.query_project = torch.nn.Linear(emb_size, emb_size)
        self.atomic_tree_emb = torch.nn.Sequential(
            PreNormLayer(tree_nfeats),
            torch.nn.Linear(tree_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

    def forward(self, constraint_features, edge_indices_0, edge_indices_1, edge_features, variable_features,
                tree_features=None, candidates=None, nb_candidates=None):

        # self.tree_to_query(tree_features, candidates)

        edge_features = self.edge_embedding(edge_features).squeeze(-1)

        # A = create_A_tensor(edge_indices_0, edge_indices_1, edge_features,
        #                     (constraint_features.size(0), variable_features.size(0)))
        edge_i, edge_w = transfer_edge(edge_indices_0, edge_indices_1, edge_features, variable_features.size(0))

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        variable_features = self.var_embedding(variable_features)
        x = torch.vstack((variable_features, constraint_features))

        x_out = self.gcn_forward(x, edge_i, edge_w)

        x = x_out[candidates]
        K = self.key_project(x)
        # V = self.value_project(x)
        atomic_f, var_changed, branch_history, pse_scores = tree_features
        batch_size = 1 if isinstance(nb_candidates, int) else len(var_changed)

        Q = self.query_project(self.atomic_tree_emb(atomic_f.view(batch_size, -1)))
        Q = Q.view(1, -1).expand(nb_candidates, -1) if isinstance(nb_candidates, int) else \
            torch.cat([q.view(1, -1).expand(nums, -1) for q, nums in zip(Q, nb_candidates)], 0)
        attention = torch.softmax((Q*K).sum(-1)/math.sqrt(K.size(-1)), dim=-1)

        # if isinstance(nb_candidates, int):
        #     Q = Q.view(1, -1).expand(nb_candidates, -1)
        #     attention = torch.softmax(Q.mm(K.transpose(-1, 0)) / math.sqrt(V.size(-1)), dim=-1).mm(V)
        # else:
        #     nb_candidates_indices = nb_candidates.cumsum(-1)
        #     attention = []
        #     for i in range(len(nb_candidates)):
        #         if i==0:
        #             i_s = 0
        #         else:
        #             i_s = nb_candidates_indices[i-1]
        #         i_e = nb_candidates_indices[i]
        #         Qi = Q[i].view(1, -1)
        #         Ki = K[i_s:i_e]
        #         Vi = V[i_s:i_e]
        #         attention.append(
        #             torch.softmax(Qi.mm(Ki.transpose(-1, 0)) / math.sqrt(Vi.size(-1)), dim=-1).mm(Vi))
        #     attention = torch.cat(attention, 0)

        # Q = Q.view(1, -1).expand(nb_candidates, -1) if isinstance(nb_candidates, int) else \
        #     torch.cat([q.view(1, -1).expand(nums, -1) for q, nums in zip(Q, nb_candidates)], 0)
        # probs = self.output(Q + K)
        # probs = self.output_module(attention)
        return attention

    def tree_to_query(self, tree_features):
        atomic_f, var_changed, branch_history, pse_scores = tree_features

        return 1

class GNNPolicy4(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19
        tree_nfeats=10

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size)

        self.key_project = nn.Linear(emb_size, emb_size, bias=False)
        self.Q_project = nn.Linear(emb_size, emb_size, bias=False)
        self.var_chg_project = nn.Linear(emb_size, emb_size)
        self.history_project = nn.Linear(emb_size, emb_size)
        self.q_weights = nn.Parameter(torch.ones(3,1))

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )
        self.atomic_tree_emb = torch.nn.Sequential(
            PreNormLayer(tree_nfeats),
            torch.nn.Linear(tree_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        self.padzero = torch.zeros(1, emb_size).to(DEVICE)
        # PSEcost EMBEDDING
        self.pse_embedding = torch.nn.Sequential(
            PreNormLayer(1),
            torch.nn.Linear(1, emb_size)
        )

    def forward(self, constraint_features, edge_indices_0, edge_indices_1, edge_features, variable_features,
                tree_features=None, candidates=None, nb_candidates=None, nb_vars=None):
        reversed_edge_indices = torch.stack([edge_indices_1, edge_indices_0], dim=0)
        edge_indices = torch.stack([edge_indices_0, edge_indices_1], dim=0)
        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        tree_feature, vars_changed, branch_history, pse_scores = tree_features

        if isinstance(nb_vars, int):
            if len(vars_changed)==0:
                var_chg_emb = self.padzero
            elif len(vars_changed)==1:
                var_chg_emb = self.var_chg_project(variable_features[vars_changed])
            else:
                var_chg_emb = self.var_chg_project(
                    variable_features.index_select(0, torch.LongTensor(vars_changed).to(variable_features.device)).mean(0)).unsqueeze(0)
            if len(branch_history)==0:
                branc_history_emb = self.padzero
            elif len(branch_history)==1:
                branc_history_emb = self.history_project(variable_features[branch_history])
            else:
                branc_history_emb = self.history_project(
                    variable_features.index_select(0, torch.LongTensor(branch_history).to(variable_features.device)).mean(0)).unsqueeze(0)

        else:
            var_chg_emb = []
            branc_history_emb = []
            nb_vars_indices = nb_vars.cumsum(-1)
            for i in range(len(nb_vars)):
                if i==0:
                    i_s = 0
                else:
                    i_s = nb_vars_indices[i-1]
                i_e = nb_vars_indices[i]
                vi = variable_features[i_s:i_e]

                if len(vars_changed[i])==0:
                    var_chg_emb.append(self.padzero)
                elif len(vars_changed[i])==1:
                    var_chg_emb.append(self.var_chg_project(vi[vars_changed[i]]))
                else:
                    var_chg_emb.append(self.var_chg_project(vi.index_select(0, torch.LongTensor(vars_changed[i]).to(vi.device)).mean(0)).unsqueeze(0))

                if len(branch_history[i])==0:
                    branc_history_emb.append(self.padzero)
                elif len(branch_history[i])==1:
                    branc_history_emb.append(self.history_project(vi[branch_history[i]]))
                else:
                    branc_history_emb.append(self.history_project(vi.index_select(0, torch.LongTensor(branch_history[i]).to(vi.device)).mean(0)).unsqueeze(0))
            var_chg_emb = torch.cat(var_chg_emb, 0)
            branc_history_emb = torch.cat(branc_history_emb, 0)

        tree_f = self.atomic_tree_emb(tree_feature.view(-1, 10))
        tree_f = self.q_weights[0]*tree_f + self.q_weights[1]*var_chg_emb + self.q_weights[2]*branc_history_emb
        Q = tree_f.view(1, -1).expand(nb_candidates, -1) if isinstance(nb_candidates, int) else \
            torch.cat([q.view(1, -1).expand(nums, -1) for q, nums in zip(tree_f, nb_candidates)], 0)
        # A final MLP on the variable features
        Q = self.Q_project(Q)
        K = self.key_project(variable_features[candidates]).squeeze(-1)
        pse_emb = self.pse_embedding(pse_scores.unsqueeze(-1))

        output = 10*torch.tanh(self.output_module(Q + K + pse_emb))

        return output.squeeze(-1)

class GNNPolicy3(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19
        tree_nfeats=10

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size)

        self.key_project = nn.Linear(emb_size, emb_size)

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )
        self.atomic_tree_emb = torch.nn.Sequential(
            PreNormLayer(tree_nfeats),
            torch.nn.Linear(tree_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

    def forward(self, constraint_features, edge_indices_0, edge_indices_1, edge_features, variable_features,
                tree_features=None, candidates=None, nb_candidates=None, nb_vars=None):
        reversed_edge_indices = torch.stack([edge_indices_1, edge_indices_0], dim=0)
        edge_indices = torch.stack([edge_indices_0, edge_indices_1], dim=0)
        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        tree_f = self.atomic_tree_emb(tree_features[0].view(-1, 10))
        Q = tree_f.view(1, -1).expand(nb_candidates, -1) if isinstance(nb_candidates, int) else \
            torch.cat([q.view(1, -1).expand(nums, -1) for q, nums in zip(tree_f, nb_candidates)], 0)
        # A final MLP on the variable features
        K = self.key_project(variable_features[candidates]).squeeze(-1)

        output = self.output_module(Q + K)

        return output.squeeze(-1)

class BipartiteGraphConvolution3(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """
    def __init__(self, emb_size=64):
        super().__init__('add')

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.edge_weight_project = nn.Sequential(
            nn.Linear(emb_size, 1),
            nn.ReLU(),
            nn.LayerNorm(1)
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        edge_features = self.feature_module_edge(edge_features)
        edge_weights = self.edge_weight_project(edge_features)
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + edge_features
                                           + edge_weights*self.feature_module_right(node_features_j))
        return output

class GNNPolicy(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size)

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices_0, edge_indices_1, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices_1, edge_indices_0], dim=0)
        edge_indices = torch.stack([edge_indices_0, edge_indices_1], dim=0)
        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """
    def __init__(self, emb_size=64):
        super().__init__('add')

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        return output


if __name__ == '__main__':
    pass