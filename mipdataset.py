import torch_geometric
from pathlib import Path
import gzip
import pickle
import torch
import numpy  as np

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = torch.FloatTensor(constraint_features)
        self.edge_index = torch.LongTensor(edge_indices.astype(np.int64))
        self.edge_attr = torch.FloatTensor(edge_features).unsqueeze(1)
        self.variable_features = torch.FloatTensor(variable_features)
        self.candidates = candidates
        self.nb_candidates = len(candidates)
        self.nb_vars = self.variable_features.size(0)
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores
        # self.A = create_A(edge_indices[0], edge_indices[1], edge_features, size_=(constraint_features.shape[0], variable_features.shape[0]))

    def __inc__(self, key, value):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample

        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
        # #         Normalize
        #         candidate_scores = (candidate_scores - candidate_scores.min()) / \
        #         (candidate_scores.max() - candidate_scores.min() + 1e-10)
        candidate_choice = torch.where(candidates == sample_action)[0][0]

        graph = BipartiteNodeData(sample_observation.row_features, sample_observation.edge_features.indices,
                                  sample_observation.edge_features.values, sample_observation.column_features,
                                  candidates, candidate_choice, candidate_scores)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.row_features.shape[0]+sample_observation.column_features.shape[0]

        return graph


class TreeNodeData(BipartiteNodeData):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, candidate_scores, tree_state, pse_scores):
        super().__init__(constraint_features, edge_indices, edge_features, variable_features,
                         candidates, candidate_choice, candidate_scores)

        self.nb_var = len(variable_features)
        self.tree_feature = torch.FloatTensor(tree_state[0])
        self.vars_changed = [int(var) for var in tree_state[1] if int(var)<self.nb_var]
        self.branch_history = [int(var) for var in tree_state[2] if int(var)<self.nb_var]
        self.pse_scores = pse_scores

        # c_ind = np.concatenate([np.where(edge_indices[1]==i.numpy()) for i in candidates],1).squeeze(0)
        # c_edge_inds = edge_indices[:, c_ind]
        # c_edge_inds[1] = np.concatenate([np.argwhere(candidates.numpy() == i_var) for i_var in c_edge_inds[1]], 1).squeeze(0)
        # self.c_edge_index = torch.LongTensor(c_edge_inds.astype(np.int64))
        # self.c_edge_attr = torch.FloatTensor(edge_features[c_ind]).unsqueeze(1)
        # self.c_variable_features = torch.FloatTensor(variable_features[candidates])
        # self.c_constraint_features =

    def __inc__(self, key, value):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class TreeDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores, tree_state, pse_scores = sample

        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        action_set = np.array(sample_action_set, dtype=np.int32)
        candidates = torch.LongTensor(action_set)
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
        pse_scores = torch.FloatTensor([pse_scores[j] for j in candidates])
        # #         Normalize
        #         candidate_scores = (candidate_scores - candidate_scores.min()) / \
        #         (candidate_scores.max() - candidate_scores.min() + 1e-10)
        candidate_choice = torch.where(candidates == sample_action)[0][0]

        # variable_features, edge_indices, edge_features = \
        #     sample_observation.column_features, sample_observation.edge_features.indices, sample_observation.edge_features.values
        # c_ind = np.concatenate([np.argwhere(edge_indices[1] == i) for i in action_set], 0).squeeze(1)
        # c_edge_inds = edge_indices[:, c_ind]
        # c_edge_inds[1] = np.concatenate([np.argwhere(action_set == i_var) for i_var in c_edge_inds[1]], 1).squeeze(0)
        # c_edge_attr = edge_features[c_ind]
        # c_var_features = variable_features[action_set]

        # graph = TreeNodeData(sample_observation.row_features, c_edge_inds,
        #                      c_edge_attr, c_var_features,
        #                           candidates, candidate_choice, candidate_scores, tree_state, pse_scores)
        graph = TreeNodeData(sample_observation.row_features, sample_observation.edge_features.indices,
                             sample_observation.edge_features.values, sample_observation.column_features,
                             candidates, candidate_choice, candidate_scores, tree_state, pse_scores)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.row_features.shape[0]+sample_observation.column_features.shape[0]

        return graph


def test_dataset():
    from tqdm import tqdm
    sample_path = "samples/setcover_tree/train"
    sample_files = [str(path) for path in Path(sample_path).glob('sample_*.pkl')]
    valid_files = sample_files[:int(0.01*len(sample_files))]
    valid_data = TreeDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=16, shuffle=False)
    for batch in tqdm(valid_loader):
        batch = batch.to(DEVICE)

        print(batch)

if __name__ == '__main__':
    from config import Config
    DEVICE = Config.DEVICE

    test_dataset()