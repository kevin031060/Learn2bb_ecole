import ecole
import torch
from pathlib import Path
from model import *
from utils import *
import numpy as np
import copy
import math

NB_EPOCHS = 10000
LEARNING_RATE = 0.0001

device_ids=range(torch.cuda.device_count())
torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
# DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

check_path = "checkpoints/setcover/20210709_2344/para_best.pt"

scip_parameters = init_params(presolve=False)
# scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(),
                                  reward_function=ecole.reward.NNodes(),
                                  information_function={"nodes": ecole.reward.NNodes().cumsum(),
                                                        "time": ecole.reward.SolvingTime().cumsum()},
                                  scip_params=scip_parameters)

checkpoint = torch.load(check_path)
model = GNNPolicy()
model.cuda()
model.load_state_dict(checkpoint)
# model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
# model.module.load_state_dict(checkpoint)

base_model = copy.deepcopy(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
instances.seed(12)
env.seed(123)
Path("checkpoints/setcover/reinf").mkdir(parents=True, exist_ok=True)


def val_net(model, val_nums = 10):
    instances_val = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    instances_val.seed(111)
    print(f"Val begin")
    costs=[]
    for instance_count, instance in zip(range(val_nums), instances_val):
        observation, action_set, _, done, info_base = env.reset(instance)
        while not done:
            with torch.no_grad():
                observation = (torch.from_numpy(observation.row_features.astype(np.float32)).cuda(),
                               torch.from_numpy(observation.edge_features.indices.astype(np.int64)).cuda()[0],
                               torch.from_numpy(observation.edge_features.indices.astype(np.int64)).cuda()[1],
                               torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).cuda(),
                               torch.from_numpy(observation.column_features.astype(np.float32)).cuda())
                logit = model(*observation)
                logit = logit[action_set.astype(np.int64)]
                prob = torch.softmax(logit, dim = -1)
                observation, action_set, reward, done, info_base = env.step(action_set[prob.argmax()])
        cost = info_base['nodes']
        costs.append(cost)
    return np.average(costs)

def train_net(log_probs, advantage):
    log_probs = torch.stack(log_probs, 1)
    # print(cost, baseline_v, ep)
    loss = advantage * log_probs.mean()
    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, 10)
    # print(advantage, grad_norms)
    optimizer.step()

for instance_count, instance in zip(range(NB_EPOCHS), instances):
    print(f"Instance: {instance_count} begin")
    # run the baseline model
    observation, action_set, _, done, info_base = env.reset(instance)
    while not done:
        with torch.no_grad():
            observation = (torch.from_numpy(observation.row_features.astype(np.float32)).cuda(),
                           torch.from_numpy(observation.edge_features.indices.astype(np.int64)).cuda()[0],
                           torch.from_numpy(observation.edge_features.indices.astype(np.int64)).cuda()[1],
                           torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).cuda(),
                           torch.from_numpy(observation.column_features.astype(np.float32)).cuda())
            logit = base_model(*observation)
            logit = logit[action_set.astype(np.int64)]
            prob = torch.softmax(logit, dim = -1)
            observation, action_set, reward, done, info_base = env.step(action_set[prob.argmax()])
    baseline_v = info_base['nodes']
    # Run the RL brancher
    observation, action_set, _, done, info = env.reset(instance)
    log_probs = []
    max_per_update = 300
    max_all_update = 3000
    ep = 0
    while not done:
        ep += 1
        with torch.set_grad_enabled(optimizer is not None):
            observation = (torch.from_numpy(observation.row_features.astype(np.float32)).cuda(),
                           torch.from_numpy(observation.edge_features.indices.astype(np.int64)).cuda()[0],
                           torch.from_numpy(observation.edge_features.indices.astype(np.int64)).cuda()[1],
                           torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).cuda(),
                           torch.from_numpy(observation.column_features.astype(np.float32)).cuda())
            logit = model(*observation)
            logit = logit[action_set.astype(np.int64)]
            prob = torch.softmax(logit, dim = -1)
            # Note that this is equivalent to what used to be called multinomial
            m = torch.distributions.Categorical(prob)
            action_ind = m.sample()
            log_prob = m.log_prob(action_ind)
            observation, action_set, reward, done, info = env.step(action_set[action_ind])

            log_probs.append(log_prob.unsqueeze(0))
            # print(ep, sep=",", end=",")
        if ep > max_all_update:
            baseline_v = info['nodes']
            break
        if ep > max_per_update:
            # compute the gradient based on current cost and baseline
            advantage = max(info['nodes'] - baseline_v, 0) / baseline_v
            train_net(log_probs, advantage)
            print(advantage, info['nodes'], baseline_v)
            log_probs = []
            # reset the baseline value to cost. The cost increase in the next iteration.
            baseline_v = max(info['nodes'], baseline_v)
            max_per_update += 300

    if len(log_probs)==0:
        continue
    advantage = (info['nodes'] - baseline_v) / baseline_v
    train_net(log_probs, advantage)
    print(advantage, info['nodes'], baseline_v)

    if instance_count % 100 == 0:
        loss_current = val_net(model, 10)
        loss_base = val_net(base_model, 10)
        print("Val:", loss_current, loss_base)
        if loss_current < loss_current:
            base_model = model
            print("!!!! Replace")
    if instance_count % 500 == 0:
        save_path = f"checkpoints/setcover/reinf/para_{instance_count}.pt"
        torch.save(model.state_dict(), save_path)




