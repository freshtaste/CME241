from typing import TypeVar, Mapping, List, Any
import numpy as np

# define generic type S and MP type
S = TypeVar('S')
A = TypeVar('A')
SSf = Mapping[S, Mapping[S, float]]
SAf = Mapping[S, Mapping[A, float]]
SASf = Mapping[S, Mapping[A, Mapping[S, float]]]

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