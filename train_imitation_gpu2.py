import torch
import torch.nn.functional as F
import torch_geometric
from pathlib import Path
import gzip
import pickle
import numpy as np
from tqdm import tqdm
from model import *
from scipy.sparse import coo_matrix
from config import Config
from utils import *
from mipdataset import *
import datetime
import time
from mipdataset import TreeDataset
DEVICE = torch.device("cuda:1")

def pretrain(model, dataloader, is_tree = False):
    """
    Pre-normalizes a model (i.e., PreNormLayer layers) over the given samples.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : torch.utils.data.DataLoader
        Dataset to use for pre-training the model.
    Return
    ------
    number of PreNormLayer layers processed.
    """
    model.pre_train_init()
    i = 0
    while True:
        for batch in tqdm(dataloader):
            batch = batch.to(DEVICE)
            if is_tree:
                tree_features = [batch.tree_feature, batch.vars_changed, batch.branch_history, batch.pse_scores]
                batched_states = (batch.constraint_features, batch.edge_index[0], batch.edge_index[1],
                                  batch.edge_attr, batch.variable_features,
                                  tree_features, batch.candidates, batch.nb_candidates, batch.nb_vars)
            else:
                # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
                batched_states = (batch.constraint_features, batch.edge_index[0], batch.edge_index[1],
                                  batch.edge_attr, batch.variable_features)

            if not model.pre_train(batched_states):
                break

        res = model.pre_train_next()
        if res is None:
            break
        else:
            layer = res

        i += 1

    return i

def process(policy, data_loader, optimizer=None, is_tree=False, device = None, top_k = None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in tqdm(data_loader):
            if device is None:
                batch = batch.to(DEVICE)
            else:
                batch = batch.to(device)

            if is_tree:
                tree_features = (batch.tree_feature, batch.vars_changed, batch.branch_history, batch.pse_scores)
                logits = policy(batch.constraint_features, batch.edge_index[0], batch.edge_index[1],
                                batch.edge_attr, batch.variable_features,
                                tree_features, batch.candidates, batch.nb_candidates, batch.nb_vars)
                logits = pad_tensor(logits, batch.nb_candidates)
            else:
                # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
                logits = policy(batch.constraint_features, batch.edge_index[0], batch.edge_index[1],
                                batch.edge_attr, batch.variable_features)
                # Index the results by the candidates, and split and pad them
                logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            #             loss = F.cross_entropy(logits, batch.candidate_choices)
            #
            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_probs = F.softmax(true_scores, dim = -1)
            predicted_log_probs = F.log_softmax(logits, dim = -1)

            kl_loss = F.kl_div(predicted_log_probs, true_probs, reduction='batchmean')
            loss_func = torch.nn.CrossEntropyLoss()
            # cpy_loss =loss_func(logits, batch.candidate_choices)
            if optimizer is not None:
                optimizer.zero_grad()
                kl_loss.backward()

                optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()

            if top_k is not None:
                kacc = []
                for k in top_k:
                    if k>logits.size(1):
                        kacc.append(0.9)
                        continue
                    pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
                    pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
                    kacc.append(
                        np.mean(np.any(pred_top_k_true_scores.cpu().numpy() == true_bestscore.cpu().numpy(), axis=1)))
                kacc = np.asarray(kacc)

            mean_loss += kl_loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            mean_kacc += kacc * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_acc, mean_kacc


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output

def train(problem = "setcover", model_name = "tree"):

    t1=time.time()

    #model = "tree" gnnm, gnn
    run_flag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    config = Config(run_flag, problem)
    save_path = config.save_path
    log_path = config.log_path
    if model_name == "gnn":
        is_tree = False
    else:
        is_tree = True

    MIPDataset = TreeDataset

    sample_path = f"samples/{problem}_tree/train"
    if problem == "indset":
        sample_path = f"/media/kevin/000B63CD00065E05/samples/{problem}_tree/train"
    # different functions
    if model_name == "tree":
        policy = GNNPolicy4()
    if model_name == "gnnm":
        policy = GNNPolicy2()
    if model_name == "gnn":
        policy = GNNPolicy()

    LEARNING_RATE = 0.001
    NB_EPOCHS = 1000
    PATIENCE = 10
    EARLY_STOPPING = 20
    batch_size = 32
    epoch_size = 312
    if problem=="auction":
        epoch_size = 312
    val_size = 200
    top_k = [3, 5, 10]
    # 1231
    seed = 12311
    rng = np.random.RandomState(seed)
    load_model = None
    # if is_tree:
    #     load_model = "checkpoints/auction/20210723_1932/auction_best_110.pt"
    if load_model is not None:
        policy.load_state_dict(torch.load(load_model))
    policy = policy.to(DEVICE)
    # policy = torch.nn.DataParallel(policy.to(DEVICE), device_ids=[0, 1])
    # policy.module.load_state_dict(torch.load(load_model))

    sample_files = [str(path) for path in Path(sample_path).glob('sample_*.pkl')]
    train_files = sample_files[:int(0.9*len(sample_files))]
    valid_files = sample_files[int(0.9*len(sample_files)):]
    pretrain_files = [f for i, f in enumerate(sample_files) if i % 20 == 0]
    pretrain_loader = torch_geometric.data.DataLoader(MIPDataset(pretrain_files), batch_size=128, shuffle=False)
    valid_data = MIPDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    best_loss = np.inf
    best_acc = -np.inf
    if load_model is not None:
        save_path = load_model.split("/")[0]
        epoch = int(load_model.split("best_")[-1][:-3])+1
        valid_loss, valid_acc, val_kacc = process(policy, valid_loader, None, is_tree=is_tree, top_k=top_k)
        print(f"Best loss:{valid_loss}, acc:{valid_acc}")
        best_acc = valid_acc
        best_loss = valid_loss
        plateau_count = 0
    else:
        epoch = 0
    while epoch<NB_EPOCHS:
        print(f"Epoch {epoch+1}")

        if epoch == 0:
            n = pretrain(model=policy, dataloader=pretrain_loader, is_tree=is_tree)
            print(f"PRETRAINED {n} LAYERS")

        # data prepare
        train_dataset = rng.choice(train_files, epoch_size * batch_size, replace=True)
        train_data = MIPDataset(train_dataset)
        train_loader = torch_geometric.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        train_loss, train_acc, train_kacc = process(policy, train_loader, optimizer, is_tree=is_tree, top_k=top_k)
        print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}, top k accuracy:", train_kacc)

        # val_dataset = np.random.choice(valid_files, val_size * batch_size, replace=False)
        valid_loss, valid_acc, val_kacc = process(policy, valid_loader, None, is_tree=is_tree, top_k=top_k)
        print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}, top k accuracy:", val_kacc)
        valid_loss = round(valid_loss, 4)
        save_path_best = Path(save_path) / f'{problem}_best_{epoch}.pt'
        if valid_loss < best_loss or valid_acc > best_acc:
            plateau_count = 0
            if valid_loss < best_loss:
                best_loss = valid_loss
            if valid_acc > best_acc:
                best_acc = valid_acc
            best_epoch = epoch
            torch.save(policy.state_dict(), save_path_best)
            print(f"  best model so far")
        else:
            plateau_count += 1
            if plateau_count % EARLY_STOPPING == 0:
                print(f"  {plateau_count} epochs without improvement, early stopping")
                torch.save(policy.state_dict(), Path(save_path) / f'{problem}_stop_{epoch}.pt')
                print(f"Cost time:{time.time()-t1}, best acc: {best_acc}")
                break
            if plateau_count % PATIENCE == 0:
                LEARNING_RATE *= 0.2
                if is_tree:
                    policy.load_state_dict(torch.load(Path(save_path) / f'{problem}_best_{best_epoch}.pt'))
                    policy = policy.to(DEVICE)
                optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
                print(f"  {plateau_count} epochs without improvement, decreasing learning rate to {LEARNING_RATE}")
        if epoch % 10 == 0:
            save_path_ = Path(save_path) / f'{problem}_{epoch}.pt'
            torch.save(policy.state_dict(), save_path_)

        if epoch % 1 == 0:
            log_values(valid_acc, valid_loss, val_kacc, epoch, log_path, problem, LEARNING_RATE)

        epoch += 1

if __name__ == '__main__':
    #
    train("setcover", "tree")
    train("setcover", "gnn")
    train("setcover", "gnnm")

    # train("auction", "tree")
    # train("auction", "gnnm")
    # train("auction", "gnn")
    # if True:
        # train("indset", "tree")
        # train("indset", "gnn")
        #train("auction", "gnnm")

        #train("location", "tree")
        # train("location", "gnn")
        # train("location", "gnnm")
        #
        # train("auction", "tree")
        # train("auction", "gnn")
        # train("auction", "gnnm")




