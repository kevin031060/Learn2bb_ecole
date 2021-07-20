import ecole
import torch
from pathlib import Path
from train_imitation import *
from utils import *
import numpy as np
from config import Config
from utils import *
import time
from train_imitation import process
from mipdataset import *
from collect_dataset import TreeObservation

DEVICE = Config.DEVICE

class SBObservation:
    def __init__(self):
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        pyscipopt_model = model.as_pyscipopt()
        if pyscipopt_model.getCurrentNode() is not None:
            if pyscipopt_model.getCurrentNode().getDepth() == 0:
                return self.strong_branching_function.extract(model, done)
            else:
                return None
        else:
            return None

def run_on_instance(policy, n_rows = 500, seed = 23, test_num = 50, is_tree=False):
    # scip_parameters = init_params(presolve=False)
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    env = ecole.environment.Branching(observation_function=(ecole.observation.NodeBipartite(), TreeObservation(), ecole.observation.Pseudocosts()),
                                      information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                                                            "time": ecole.reward.SolvingTime().cumsum()},
                                      scip_params=scip_parameters)

    instances = ecole.instance.SetCoverGenerator(n_rows=n_rows, n_cols=1000, density=0.05)
    instances.seed(seed)
    env.seed(seed)

    nodes = []
    solving_time = []

    for instance_count, instance in zip(range(test_num), instances):
        # Run the GNN brancher
        observation, action_set, _, done, info = env.reset(instance)
        node_obs, tree_obs, pse_score = observation
        i=0
        while not done:
            # if i == 0:
            #     action = action_set[SBscore[action_set].argmax()]
            #     observation, action_set, _, done, info = env.step(action)
            #     node_obs, tree_obs, _ = observation
            #     i += 1
            #     continue

            with torch.no_grad():
                observation = (torch.from_numpy(node_obs.row_features.astype(np.float32)).to(DEVICE),
                               torch.from_numpy(node_obs.edge_features.indices.astype(np.int64)).to(DEVICE)[0],
                               torch.from_numpy(node_obs.edge_features.indices.astype(np.int64)).to(DEVICE)[1],
                               torch.from_numpy(node_obs.edge_features.values.astype(np.float32)).view(-1, 1).to(DEVICE),
                               torch.from_numpy(node_obs.column_features.astype(np.float32)).to(DEVICE))
                atomic_f = tree_obs[0]
                atomic_f.append(len(action_set)/node_obs.column_features.shape[0])
                pse_scores = torch.FloatTensor([pse_score[j] for j in action_set])
                tree_features = ([torch.FloatTensor(atomic_f).to(DEVICE),
                                  tree_obs[1], tree_obs[2], pse_scores.to(DEVICE)], )
                others = (torch.from_numpy(action_set.astype(np.int64)).to(DEVICE),
                          len(action_set))
                if is_tree:
                    observation = observation + tree_features + others
                    logits = policy(*observation)
                    action = action_set[logits.argmax().cpu().numpy()]
                else:
                    logits = policy(*observation)
                    action = action_set[logits[action_set.astype(np.int64)].argmax()]

                observation, action_set, _, done, info = env.step(action)
                node_obs, tree_obs, _ = observation

        nodes.append(info['nb_nodes'])
        solving_time.append(info['time'])
        print(f"Instance {instance_count: >3} | GNN  nb nodes    {int(info['nb_nodes']): >4d}  | GNN  time   {info['time']: >6.2f} ")
    return nodes, solving_time


def run_on_default_instance(n_rows = 500, seed = 23, test_num = 50):
    # scip_parameters = init_params(presolve=False)
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    default_env = ecole.environment.Configuring(observation_function=None,
                                                    information_function={"nb_nodes": ecole.reward.NNodes(),
                                                                          "time": ecole.reward.SolvingTime()},
                                                    scip_params=scip_parameters)

    instances = ecole.instance.SetCoverGenerator(n_rows=n_rows, n_cols=1000, density=0.05)
    instances.seed(seed)
    default_env.seed(seed)

    nodes = []
    solving_time = []

    for instance_count, instance in zip(range(test_num), instances):
        # Run the GNN brancher
        default_env.reset(instance)
        _, _, _, _, info = default_env.step({})

        nodes.append(info['nb_nodes'])
        solving_time.append(info['time'])
        print(f"Instance {instance_count: >3} | SCIP  nb nodes    {int(info['nb_nodes']): >4d}  | SCIP  time   {info['time']: >6.2f} ")
    return nodes, solving_time


def run_on_SB_instance(n_rows = 500, seed = 23, test_num = 50, is_SB=False):
    # scip_parameters = init_params(presolve=False)
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    env = ecole.environment.Branching(observation_function=(ecole.observation.StrongBranchingScores() if is_SB else ecole.observation.Pseudocosts()),
                                      information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                                                            "time": ecole.reward.SolvingTime().cumsum()},
                                      scip_params=scip_parameters)

    instances = ecole.instance.SetCoverGenerator(n_rows=n_rows, n_cols=1000, density=0.05)
    instances.seed(seed)
    env.seed(seed)

    nodes = []
    solving_time = []

    for instance_count, instance in zip(range(test_num), instances):
        # Run the GNN brancher
        scores, action_set, _, done, info = env.reset(instance)

        i=0
        while not done:
            action = action_set[scores[action_set].argmax()]

            scores, action_set, _, done, info = env.step(action)

        nodes.append(info['nb_nodes'])
        solving_time.append(info['time'])
        print(f"Instance {instance_count: >3} | GNN  nb nodes    {int(info['nb_nodes']): >4d}  | GNN  time   {info['time']: >6.2f} ")

    scores_name = "SB scores" if is_SB else "PSB scores"
    print(f"{scores_name}, time:{np.mean(solving_time)}, nodes: {np.mean(nodes)}")
    print(nodes)
    print(solving_time)
    return nodes, solving_time


def compare_sovling_statistic(n_rows = 500, path1=None, path2=None):

    # n_rows = 2000


    # policy = GNNPointerPolicy()
    # policy = policy.to(DEVICE)
    # # check_path = "checkpoints/setcover/20210715_2047/para_best_84.pt"
    # check_path = "checkpoints/setcover/20210716_1700/setcover_best_168.pt"
    # policy.load_state_dict(torch.load(check_path))
    #
    # nodes_p, times_p = run_on_instance(policy, n_rows=n_rows, is_tree=True)

    policy_base = GNNPolicy()
    # check_path = "checkpoints/setcover/para_best.pt"
    check_path = "checkpoints/setcover/20210717_0033/setcover_best_99.pt"
    policy_base = policy_base.to(DEVICE)
    policy_base.load_state_dict(torch.load(check_path))

    policy = GNNPolicy3()
    # check_path = "checkpoints/setcover/20210713_1516/para_best_73.pt"
    check_path = "checkpoints/setcover/20210718_0048/setcover_best_295.pt"
    check_path = "checkpoints/setcover/20210719_1020/setcover_best_39.pt"
    check_path = path1
    policy = policy.to(DEVICE)
    policy.load_state_dict(torch.load(check_path))

    nodes_m, times_m = run_on_instance(policy, n_rows=n_rows, is_tree=True)
    print(f"matrix: time:{np.mean(times_m)}, node:{np.mean(nodes_m)}")



    nodes_gnn, times_gnn = run_on_instance(policy_base, n_rows=n_rows)
    print(f"Gasse: time:{np.mean(times_gnn)}, node:{np.mean(nodes_gnn)}")


    nodes_scip, times_scip = run_on_default_instance(n_rows=n_rows)
    print(f"SCIP: time:{np.mean(times_scip)}, node:{np.mean(nodes_scip)}")


    # print(f"pointer: time:{np.mean(times_p)}, node:{np.mean(nodes_p)}")
    print(f"matrix: time:{np.mean(times_m)}, node:{np.mean(nodes_m)}")
    print(f"Gasse: time:{np.mean(times_gnn)}, node:{np.mean(nodes_gnn)}")
    print(f"SCIP: time:{np.mean(times_scip)}, node:{np.mean(nodes_scip)}")
    # print(nodes_p)
    print(nodes_m)
    print(nodes_gnn)
    print(nodes_scip)
    # print(times_p)
    print(times_m)
    print(times_gnn)
    print(times_scip)


def test_show():
    check_path = "checkpoints/setcover/20210713_1516/para_best_73.pt"
    scip_parameters = init_params(presolve=False)
    # scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(),
                                      information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                                                            "time": ecole.reward.SolvingTime().cumsum()},
                                      scip_params=scip_parameters)
    default_env = ecole.environment.Configuring(observation_function=None,
                                                information_function={"nb_nodes": ecole.reward.NNodes(),
                                                                      "time": ecole.reward.SolvingTime()},
                                                scip_params=scip_parameters)

    checkpoint = torch.load(check_path)
    policy = GNNPolicyMatrix()
    policy = policy.to(DEVICE)
    policy.load_state_dict(checkpoint)

    instances = ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=1000, density=0.05)
    instances.seed(123)
    env.seed(123)
    for instance_count, instance in zip(range(50), instances):
        # Run the GNN brancher
        observation, action_set, _, done, info = env.reset(instance)

        while not done:
            with torch.no_grad():
                observation = (torch.from_numpy(observation.row_features.astype(np.float32)).to(DEVICE),
                               torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(DEVICE)[0],
                               torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(DEVICE)[1],
                               torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(DEVICE),
                               torch.from_numpy(observation.column_features.astype(np.float32)).to(DEVICE))
                logits = policy(*observation)
                action = action_set[logits[action_set.astype(np.int64)].argmax()]
                observation, action_set, _, done, info = env.step(action)

        # Run SCIP's default brancher
        default_env.reset(instance)
        _, _, _, _, default_info = default_env.step({})

        print(f"Instance {instance_count: >3} | SCIP nb nodes    {int(default_info['nb_nodes']): >4d}  | SCIP time   {default_info['time']: >6.2f} ")
        print(f"             | GNN  nb nodes    {int(info['nb_nodes']): >4d}  | GNN  time   {info['time']: >6.2f} ")
        print(f"             | Gain         {100*(1-info['nb_nodes']/default_info['nb_nodes']): >8.2f}% | Gain      {100*(1-info['time']/default_info['time']): >8.2f}%")



def test_default():

    # scip_parameters = init_params(presolve=False)
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    default_env = ecole.environment.Configuring(observation_function=None,
                                                information_function={"nb_nodes": ecole.reward.NNodes(),
                                                                      "time": ecole.reward.SolvingTime()},
                                                scip_params=scip_parameters)

    instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    instances.seed(123)
    default_env.seed(123)
    for instance_count, instance in zip(range(10), instances):
        # Run SCIP's default brancher
        default_env.reset(instance)
        _, _, _, _, default_info = default_env.step({})

        print(f"Instance {instance_count: >3} | SCIP nb nodes    {int(default_info['nb_nodes']): >4d}  | SCIP time   {default_info['time']: >6.2f} ")

class TestReward:

    def __init__(self):
        pass

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        # Unconditionally getting reward as reward_funcition.extract may have side effects
        pyscipopt_model = model.as_pyscipopt()
        print(0)
        return pyscipopt_model.getCurrentNode()

def test_ecole():
    instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000)
    instances.seed(123)
    # We can pass custom SCIP parameters easily
    scip_parameters = init_params(presolve=False, disable_all_h= True)
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}
    # scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}
    # Note how we can tuple observation functions to return complex state information
    pRewardF = TestReward()
    env = ecole.environment.Branching(observation_function=(ecole.observation.StrongBranchingScores(),
                                                            ecole.observation.NodeBipartite(),
                                                            ecole.observation.Khalil2016()),
                                      information_function={
                                          "nb_nodes": ecole.reward.NNodes().cumsum(),
                                          "time": ecole.reward.SolvingTime().cumsum(),
                                          "P":ecole.reward.PrimalIntegral(),
                                          "D":ecole.reward.DualIntegral(),
                                          "PD":ecole.reward.PrimalDualIntegral(),
                                          "lp":ecole.reward.LpIterations()
                                      },
                                      reward_function=pRewardF,
                                      scip_params=scip_parameters)

    # This will seed the environment for reproducibility
    env.seed(123)
    observation, action_set, reward, done, info = env.reset(next(instances))
    print(info, reward)
    # (scores, scores_are_expert), node_observation = observation
    # print(node_observation.edge_features.shape, node_observation.row_features.shape, node_observation.column_features.shape)
    while not done:
        scores, node_observation, k_obs = observation
        action = action_set[scores[action_set].argmax()]
        observation, action_set, reward, done, info = env.step(action)
        print(info)


def test_accuracy(is_tree = True, path1=None):
    t1 = time.time()
    batch_size = 16
    val_size = 100
    sample_path = "samples/setcover_tree/train"
    MIPDataset = TreeDataset

    if is_tree:
        policy = GNNPolicy3()
        check_path = "checkpoints/setcover/20210719_1020/setcover_best_39.pt"
        # check_path = "checkpoints/setcover/20210715_2047/para_best_84.pt"
        check_path = path1
    else:
        policy = GNNPolicy()
        check_path = "checkpoints/setcover/20210717_0033/setcover_best_99.pt"
    sample_files = [str(path) for path in Path(sample_path).glob('sample_*.pkl')]

    valid_files = sample_files[int(0.9*len(sample_files)):]

    checkpoint = torch.load(check_path)

    policy = policy.to(DEVICE)
    policy.load_state_dict(checkpoint)

    # val_dataset = np.random.choice(valid_files, val_size * batch_size, replace=False)
    valid_data = MIPDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    valid_loss, valid_acc, mean_kacc = process(policy, valid_loader, None, is_tree=is_tree, device=DEVICE, top_k = [3,5,10])
    print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}, top k accuracy:", mean_kacc)
    print("Cost time: ", time.time() - t1)

if __name__ == '__main__':
    # 60
    check = "checkpoints/setcover/20210719_1020/setcover_best_198.pt"
    test_accuracy(is_tree=False)
    test_accuracy(is_tree=True, path1=check)
    compare_sovling_statistic(n_rows=500, path1=check)
    compare_sovling_statistic(n_rows=1000, path1=check)
    # test_default()
    # test_ecole()