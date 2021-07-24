import math
import torch
import numpy as np
from config import Config
from tensorboard_logger import Logger as TbLogger


import os
def log_values(val_accuracy, val_loss, val_kacc, epoch, log_path, problem, LEARNING_RATE):

    with open(os.path.join(log_path, f'{problem}_trace.txt'), 'a+') as f:
        f.writelines(f'epoch: {epoch}, val_accuracy: {val_accuracy}, '
                     f'val_loss: {val_loss}, kacc: {val_kacc[0]}, {val_kacc[1]}, {val_kacc[2]}'
                     + f' lr{LEARNING_RATE}'+ '\r\n')
    # Log values to tensorboard
    tb_logger = TbLogger(log_path)
    tb_logger.log_value('val_accuracy', val_accuracy, epoch)

    tb_logger.log_value('val_loss', val_loss, epoch)


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x

def to_np(t):
    return t.cpu().detach().numpy()




class SCIPParam:
    def __init__(self, init_param=None):
        if init_param is None:
            init_param = {}
        self.params = init_param

    def setIntParam(self, key, value):
        self.params[key] = value

    def getParam(self):
        return self.params



def init_params(init_param = None, presolve = False, change_default_settings = False, disable_all_h = False):
    """
    :param model: scip.Model(), model instantiation
    :param scip_limits: dict, specifying SCIP parameter limits
    :param scip_params: dict, specifying SCIP parameter setting
    :return: -
        Initialize SCIP parameters for the model.
    """
    model = SCIPParam(init_param)
    model.setIntParam('display/verblevel', 0)

    # limits
    model.setIntParam('limits/nodes', -1)
    model.setIntParam('limits/time', 3600.0)

    if not presolve:
        # disable presolve and cuts (enabled in default)
        model.setIntParam('presolving/maxrounds', 0)       # 0: off, -1: unlimited
        model.setIntParam('separating/maxrounds', 0)       # 0 to disable local separation
        model.setIntParam('separating/maxroundsroot', 0)   # 0 to disable root separation
        model.setIntParam('presolving/maxrestarts', 0)

    if change_default_settings:
        scip_params = {'heuristics': False,        # enable all primal heuristics
                      'cutoff': True,             # provide cutoff (value needs to be passed to the environment)
                      'conflict_usesb': False,    # use SB conflict analysis
                      'probing_bounds': False,    # use probing bounds identified during SB
                      'checksol': False,
                      'reevalage': 0,
                      }
    else:
        scip_params ={
            'heuristics': True,
            'cutoff': False,
            'conflict_usesb': True,
            'probing_bounds': True,
            'checksol': True,
            'reevalage': 10,
        }


    # # disable reoptimization (as in default)
    # model.setIntParam('reoptimization/enable', False)
    #
    # # cutoff value is eventually set in env.run_episode
    # # other parameters to be disabled in 'sandbox' setting
    # model.setIntParam('conflict/usesb', scip_params['conflict_usesb'])
    # model.setIntParam('branching/fullstrong/probingbounds', scip_params['probing_bounds'])
    # model.setIntParam('branching/relpscost/probingbounds', scip_params['probing_bounds'])
    # model.setIntParam('branching/checksol', scip_params['checksol'])
    # model.setIntParam('branching/fullstrong/reevalage', scip_params['reevalage'])

    # primal heuristics (54 total, 14 of which are disabled in default setting as well)
    if not scip_params['heuristics']:
        model.setIntParam('heuristics/actconsdiving/freq', -1)          # disabled at default
        model.setIntParam('heuristics/bound/freq', -1)                  # disabled at default
        model.setIntParam('heuristics/clique/freq', -1)
        model.setIntParam('heuristics/coefdiving/freq', -1)
        model.setIntParam('heuristics/completesol/freq', -1)
        model.setIntParam('heuristics/conflictdiving/freq', -1)         # disabled at default
        model.setIntParam('heuristics/crossover/freq', -1)
        model.setIntParam('heuristics/dins/freq', -1)                   # disabled at default
        model.setIntParam('heuristics/distributiondiving/freq', -1)
        # model.setIntParam('heuristics/dualval/freq', -1)                # disabled at default
        model.setIntParam('heuristics/farkasdiving/freq', -1)
        model.setIntParam('heuristics/feaspump/freq', -1)
        model.setIntParam('heuristics/fixandinfer/freq', -1)            # disabled at default
        model.setIntParam('heuristics/fracdiving/freq', -1)
        model.setIntParam('heuristics/gins/freq', -1)
        model.setIntParam('heuristics/guideddiving/freq', -1)
        model.setIntParam('heuristics/zeroobj/freq', -1)                # disabled at default
        model.setIntParam('heuristics/indicator/freq', -1)
        model.setIntParam('heuristics/intdiving/freq', -1)              # disabled at default
        model.setIntParam('heuristics/intshifting/freq', -1)
        model.setIntParam('heuristics/linesearchdiving/freq', -1)
        model.setIntParam('heuristics/localbranching/freq', -1)         # disabled at default
        model.setIntParam('heuristics/locks/freq', -1)
        model.setIntParam('heuristics/lpface/freq', -1)
        model.setIntParam('heuristics/alns/freq', -1)
        model.setIntParam('heuristics/nlpdiving/freq', -1)
        model.setIntParam('heuristics/mutation/freq', -1)               # disabled at default
        model.setIntParam('heuristics/multistart/freq', -1)
        model.setIntParam('heuristics/mpec/freq', -1)
        model.setIntParam('heuristics/objpscostdiving/freq', -1)
        model.setIntParam('heuristics/octane/freq', -1)                 # disabled at default
        model.setIntParam('heuristics/ofins/freq', -1)
        model.setIntParam('heuristics/oneopt/freq', -1)
        model.setIntParam('heuristics/proximity/freq', -1)              # disabled at default
        model.setIntParam('heuristics/pscostdiving/freq', -1)
        model.setIntParam('heuristics/randrounding/freq', -1)
        model.setIntParam('heuristics/rens/freq', -1)
        model.setIntParam('heuristics/reoptsols/freq', -1)
        model.setIntParam('heuristics/repair/freq', -1)                 # disabled at default
        model.setIntParam('heuristics/rins/freq', -1)
        model.setIntParam('heuristics/rootsoldiving/freq', -1)
        model.setIntParam('heuristics/rounding/freq', -1)
        model.setIntParam('heuristics/shiftandpropagate/freq', -1)
        model.setIntParam('heuristics/shifting/freq', -1)
        model.setIntParam('heuristics/simplerounding/freq', -1)
        model.setIntParam('heuristics/subnlp/freq', -1)
        model.setIntParam('heuristics/trivial/freq', -1)
        model.setIntParam('heuristics/trivialnegation/freq', -1)
        model.setIntParam('heuristics/trysol/freq', -1)
        model.setIntParam('heuristics/twoopt/freq', -1)                 # disabled at default
        model.setIntParam('heuristics/undercover/freq', -1)
        model.setIntParam('heuristics/vbounds/freq', -1)
        model.setIntParam('heuristics/veclendiving/freq', -1)
        model.setIntParam('heuristics/zirounding/freq', -1)
    if disable_all_h:
        model.setIntParam('heuristics/actconsdiving/freq', 0)          # disabled at default
        model.setIntParam('heuristics/bound/freq', 0)                  # disabled at default
        model.setIntParam('heuristics/clique/freq', 0)
        model.setIntParam('heuristics/coefdiving/freq', 0)
        model.setIntParam('heuristics/completesol/freq', 0)
        model.setIntParam('heuristics/conflictdiving/freq', 0)         # disabled at default
        model.setIntParam('heuristics/crossover/freq', 0)
        model.setIntParam('heuristics/dins/freq', 0)                   # disabled at default
        model.setIntParam('heuristics/distributiondiving/freq', 0)
        # model.setIntParam('heuristics/dualval/freq', 0)                # disabled at default
        model.setIntParam('heuristics/farkasdiving/freq', 0)
        model.setIntParam('heuristics/feaspump/freq', 0)
        model.setIntParam('heuristics/fixandinfer/freq', 0)            # disabled at default
        model.setIntParam('heuristics/fracdiving/freq', 0)
        model.setIntParam('heuristics/gins/freq', 0)
        model.setIntParam('heuristics/guideddiving/freq', 0)
        model.setIntParam('heuristics/zeroobj/freq', 0)                # disabled at default
        model.setIntParam('heuristics/indicator/freq', 0)
        model.setIntParam('heuristics/intdiving/freq', 0)              # disabled at default
        model.setIntParam('heuristics/intshifting/freq', 0)
        model.setIntParam('heuristics/linesearchdiving/freq', 0)
        model.setIntParam('heuristics/localbranching/freq', 0)         # disabled at default
        model.setIntParam('heuristics/locks/freq', 0)
        model.setIntParam('heuristics/lpface/freq', 0)
        model.setIntParam('heuristics/alns/freq', 0)
        model.setIntParam('heuristics/nlpdiving/freq', 0)
        model.setIntParam('heuristics/mutation/freq', 0)               # disabled at default
        model.setIntParam('heuristics/multistart/freq', 0)
        model.setIntParam('heuristics/mpec/freq', 0)
        model.setIntParam('heuristics/objpscostdiving/freq', 0)
        model.setIntParam('heuristics/octane/freq', 0)                 # disabled at default
        model.setIntParam('heuristics/ofins/freq', 0)
        model.setIntParam('heuristics/oneopt/freq', 0)
        model.setIntParam('heuristics/proximity/freq', 0)              # disabled at default
        model.setIntParam('heuristics/pscostdiving/freq', 0)
        model.setIntParam('heuristics/randrounding/freq', 0)
        model.setIntParam('heuristics/rens/freq', 0)
        model.setIntParam('heuristics/reoptsols/freq', 0)
        # model.setIntParam('heuristics/repair/freq', 0)                 # disabled at default
        model.setIntParam('heuristics/rins/freq', 0)
        model.setIntParam('heuristics/rootsoldiving/freq', 0)
        model.setIntParam('heuristics/rounding/freq', 0)
        model.setIntParam('heuristics/shiftandpropagate/freq', 0)
        model.setIntParam('heuristics/shifting/freq', 0)
        model.setIntParam('heuristics/simplerounding/freq', 0)
        model.setIntParam('heuristics/subnlp/freq', 0)
        model.setIntParam('heuristics/trivial/freq', 0)
        model.setIntParam('heuristics/trivialnegation/freq', 0)
        model.setIntParam('heuristics/trysol/freq', 0)
        model.setIntParam('heuristics/twoopt/freq', 0)                 # disabled at default
        model.setIntParam('heuristics/undercover/freq', 0)
        model.setIntParam('heuristics/vbounds/freq', 0)
        model.setIntParam('heuristics/veclendiving/freq', 0)
        model.setIntParam('heuristics/zirounding/freq', 0)
    return model.getParam()

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

if __name__ == '__main__':
    print(init_params().__len__())