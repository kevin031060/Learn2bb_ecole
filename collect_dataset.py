import gzip
import pickle
import numpy as np
import ecole
from pathlib import Path
from utils import *
import os
import multiprocessing as mp
import glob
import shutil
from scipy.sparse import coo_matrix
import time


def create_A(row, col, data, size_):
    num_row = size_[0]
    num_col = size_[1]
    row_m1, col_m1, data_m1 = np.arange(num_row), np.arange(num_row), np.ones(num_row)
    row_m2, col_m2 = row + num_row, col
    row_m3, col_m3 = col, row + num_row
    row_m4, col_m4, data_m4 = np.arange(num_col)+num_row, np.arange(num_col)+num_row, np.ones(num_col)
    row_A = np.concatenate((row_m1, row_m2, row_m3, row_m4))
    col_A = np.concatenate((col_m1, col_m2, col_m3, col_m4))
    data_A = np.concatenate((data_m1, data, data, data_m4))
    print(row_A.max(), num_row+num_col)
    return coo_matrix((data_A, (row_A, col_A)), shape=(num_row+num_col, num_row+num_col)).toarray()


class TreeObservation:
    def __init__(self):
        pass

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        pyscipopt_model = model.as_pyscipopt()
        depth = pyscipopt_model.getDepth() / 10
        gap = pyscipopt_model.getGap() * 3
        infeasibleLeaves = pyscipopt_model.getNInfeasibleLeaves() / max(pyscipopt_model.getNLeaves(), 0.1)
        feasibleLeaves = pyscipopt_model.getNFeasibleLeaves() / max(pyscipopt_model.getNLeaves(), 0.1)

        obj_primal_b = self.cal_distance(pyscipopt_model.getLPObjVal(), pyscipopt_model.getPrimalbound())
        obj_dual_b = self.cal_distance(pyscipopt_model.getLPObjVal(), pyscipopt_model.getDualbound())
        obj_primaldual_b = np.abs(pyscipopt_model.getLPObjVal() - pyscipopt_model.getPrimalbound()) \
                           / max(np.abs(pyscipopt_model.getPrimalbound() - pyscipopt_model.getDualbound()), 0.1)

        root_dual_b = self.cal_distance(pyscipopt_model.getDualboundRoot(), pyscipopt_model.getDualbound())
        root_dual_obj_b = self.cal_distance(pyscipopt_model.getLPObjVal(), pyscipopt_model.getDualboundRoot())

        vars_changed = []
        branch_history = []
        if pyscipopt_model.getCurrentNode() is not None:
            if pyscipopt_model.getCurrentNode().getDomchg() is not None:
                # variable bound changes when branching to obtain this node
                changes = pyscipopt_model.getCurrentNode().getDomchg().getBoundchgs()
                vars_changed = [int(change.getVar().name.split("_")[-1]) for change in changes]
            # branch history
            node_ = pyscipopt_model.getCurrentNode()
            while node_.getParentBranchings() is not None:
                branch_history.append(int(node_.getParentBranchings()[0][0].name.split("_")[-1]))
                node_ = node_.getParent()
                if node_ is None:
                    break

        features = [depth, gap, infeasibleLeaves, feasibleLeaves, obj_primal_b, obj_dual_b,
                    obj_primaldual_b, root_dual_obj_b, root_dual_b]
        return (features, vars_changed, branch_history)

    def cal_distance(self, v1, v2):
        return np.abs(v1 - v2) / np.max([np.abs(v1), np.abs(v2), 0.1])


class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    """
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()
        self.tree_function = TreeObservation()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done),
                    self.pseudocosts_function.extract(model, done), self.tree_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), self.pseudocosts_function.extract(model, done), None, False)




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


def make_samples(in_queue, out_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """
    while True:
        episode, instance, seed, time_limit, outdir, rng = in_queue.get()
        out_queue.put({
            "type":'start',
            "episode":episode,
            "seed": seed
        })

        if instance == "setcover":
            instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000)
        elif instance == "auction":
            instances = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500)
        elif instance == "location":
            instances = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100)
        elif instance == "indset":
            instances = ecole.instance.IndependentSetGenerator()
        else:
            instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000)
        instances.seed(seed)
        # We can pass custom SCIP parameters easily
        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}
        # Note how we can tuple observation functions to return complex state information
        env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=0.05),
                                                                ecole.observation.NodeBipartite()),
                                          scip_params=scip_parameters)
        # This will seed the environment for reproducibility
        env.seed(seed)
        sample_counter = 0
        filenames = []
        # start
        observation, action_set, _, done, _ = env.reset(next(instances))
        while not done:
            (scores, pse_scores, tree_state, scores_are_expert), node_observation = observation

            action = action_set[scores[action_set].argmax()]
            # Only save samples if they are coming from the expert (strong branching)
            if scores_are_expert:
                sample_counter += 1
                if tree_state is not None:
                    tree_state[0].append(len(action_set)/len(scores))
                data = [node_observation, action, action_set, scores, tree_state, pse_scores]
                filename = f'{outdir}/sample_{episode}_{sample_counter}.pkl'
                filenames.append(filename)
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)
            observation, action_set, _, done, _ = env.step(action)

        out_queue.put({
                "type": "done",
                "episode": episode,
                "seed": seed,
                "filenames":filenames,
                "nnodes":len(filenames),
            })



def send_orders(orders_queue, instance, seed, time_limit, outdir, start_episode):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Input queue from which orders are received.
    instances : name (setcover)
        name of instances which are solved by SCIP to collect data
    seed : int
        initial seed to insitalize random number generator with
    time_limit : int
        maximum time for which to solve an instance while collecting data
    outdir : str
        directory where to save data
    start_episode : int
        episode to resume data collection. It is used if the data collection process was stopped earlier for some reason.
    """
    rng = np.random.RandomState(seed)
    episode = 0
    while True:
        seed = rng.randint(2**32)
        # already processed; for a broken process; for root dataset to not repeat instances and seed
        if episode <= start_episode:
            episode += 1
            continue

        orders_queue.put([episode, instance, seed, time_limit, outdir, rng])
        episode += 1



def collect_samples(instance, outdir, rng, n_samples, n_jobs, time_limit):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    instances : list
        filepaths of instances which will be solved to collect data
    outdir : str
        directory where to save data
    rng : np.random.RandomState
        random number generator
    n_samples : int
        total number of samples to collect.
    n_jobs : int
        number of CPUs to utilize or number of instances to solve in parallel.
    time_limit : int
        maximum time for which to solve an instance while collecting data
    """
    os.makedirs(outdir, exist_ok=True)

    # start workers
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()
    workers = []
    for i in range(n_jobs):
        p = mp.Process(
            target=make_samples,
            args=(orders_queue, answers_queue),
            daemon=True)
        workers.append(p)
        p.start()

    # dir to keep samples temporarily; helps keep a prefect count
    tmp_samples_dir = f'{outdir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # if the process breaks due to some reason, resume from this last_episode.
    existing_samples = glob.glob(f"{outdir}/*.pkl")
    last_episode, last_i = -1, 0
    if existing_samples:
        last_episode = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[1]) for x in existing_samples) # episode is 2nd last
        last_i = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[-1]) for x in existing_samples) # sample number is the last

    # start dispatcher
    dispatcher = mp.Process(
        target=send_orders,
        args=(orders_queue, instance, rng.randint(2**32), time_limit, tmp_samples_dir, last_episode),
        daemon=True)
    dispatcher.start()

    i = last_i # for a broken process
    in_buffer = 0
    t1 = time.time()
    while i <= n_samples:
        sample = answers_queue.get()

        if sample['type'] == 'start':
            in_buffer += 1

        if sample['type'] == 'done':
            for filename in sample['filenames']:
                x = filename.split('/')[-1].split(".pkl")[0]
                os.rename(filename, f"{outdir}/{x}_{i}.pkl")
                i+=1
                print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

                if  i == n_samples:
                    # early stop dispatcher (hard)
                    if dispatcher.is_alive():
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...")
                    break

        if not dispatcher.is_alive():
            break

    # stop all workers (hard)
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)

def run(name = "auction"):
    # 21.50
    seed = 123
    time_limit = 3600
    train_size = 160000
    n_jobs = 16
    out_dir = f"samples/{name}_tree"
    rng = np.random.RandomState(seed + 1)
    t1=time.time()
    # collect
    collect_samples(name, out_dir +"/train", rng, train_size, n_jobs, time_limit)
    print(time.time()-t1)

if __name__ == '__main__':
    run("location")
    run("indset")