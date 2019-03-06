from typing import TypeVar, Mapping, List, Any, Callable, Tuple, Set
import numpy as np
from scipy.stats import rv_discrete

# define generic type S and MP type
S = TypeVar('S')
A = TypeVar('A')
SSf = Mapping[S, Mapping[S, float]]
SAf = Mapping[S, Mapping[A, float]]
SASf = Mapping[S, Mapping[A, Mapping[S, float]]]
SASTff = Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]

def get_all_states(d: Mapping[S, Any]) -> List[S]:
    return list(d.keys())

def get_transition_matrix(tr: SSf) -> np.ndarray:
    s_list = get_all_states(tr)
    mat = np.zeros((len(s_list), len(s_list)))
    for s_i in s_list:
        for s_j in list(tr[s_i].keys()):
            mat[s_list.index(s_i),s_list.index(s_j)] = tr[s_i][s_j]
    return mat

def verify_mp(tr: SSf) -> bool:
    Sf = [dic for dic in tr.values()]
    row_sum = [sum([val for val in item.values()]) for item in Sf]
    for each_row in row_sum:
        if np.abs(each_row - 1) > 1e-8:
            return False
    return True

def verify_mdp(tr: SASf) -> bool:
    row_sum = [sum(list(sf.values())) for dic in tr.values() for sf in dic.values()]
    for each_row in row_sum:
        if np.abs(each_row - 1) > 1e-8:
            return False
    return True

def get_actions_for_states(mdp_data: Mapping[S, Mapping[A, Any]])\
        -> Mapping[S, Set[A]]:
    return {k: set(v.keys()) for k, v in mdp_data.items()}

def get_rv_gen_func_single(prob_dict: Mapping[Any, float])\
        -> Callable[[], S]:
    outcomes, probabilities = zip(*prob_dict.items())
    rvd = rv_discrete(values=(range(len(outcomes)), probabilities))
    
    return lambda rvd=rvd, outcomes=outcomes: outcomes[rvd.rvs(size=1)[0]]

def get_state_reward_gen_func(
    prob_dict: Mapping[S, float],
    rew_dict: Mapping[S, float]
) -> Callable[[], Tuple[S, float]]:
    gf = get_rv_gen_func_single(prob_dict)
    
    def ret_func(gf=gf, rew_dict=rew_dict) -> Tuple[S, float]:
        state_outcome = gf()
        reward_outcome = rew_dict[state_outcome]
        return state_outcome, reward_outcome

    return ret_func

def get_state_reward_gen_dict(tr: SASf, rr: SASf) \
        -> Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]:
    return {s: {a: get_state_reward_gen_func(tr[s][a], rr[s][a])
                for a, _ in v.items()}
            for s, v in rr.items()}

def mdp_refined_split_info(info: SASTff) -> Tuple[SASf, SASf, SAf]:
    tr = {s: {a: {s1: v2[0] for s1,v2 in v1.items()} for a,v1 in v.items()} 
                for s,v in info.items()}
    rr = {s: {a: {s1: v2[1] for s1,v2 in v1.items()} for a,v1 in v.items()} 
                for s,v in info.items()}
    state_reward = {s: {a: sum([v2[0]*v2[1] for s1,v2 in v1.items()]) for a,v1 in v.items()} 
                for s,v in info.items()}
    return tr, rr, state_reward

def get_expected_action_value(action_qv: Mapping[A, float], epsilon: float) -> float:
    _, val_opt = max(action_qv.items(), key = lambda l:l[1])
    m = len(action_qv.keys())
    return sum([val*epsilon/m for val in action_qv.values()]) \
               + val_opt * (1 - epsilon)
               
def get_epsilon_greedy_action(action_qv: Mapping[A, float], epsilon: float) -> float:
    action_opt, val_opt = max(action_qv.items(), key = lambda l:l[1])
    m = len(action_qv.keys())
    prob_dict = {a: epsilon/m + 1 - epsilon if a == action_opt else epsilon/m\
                 for a,v in action_qv.items()}
    gf = get_rv_gen_func_single(prob_dict)
    
    return gf()