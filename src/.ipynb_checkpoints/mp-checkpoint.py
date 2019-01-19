from typing import Mapping, Set, Generic, Sequence
from my_funcs import get_all_states, get_transition_matrix, verify_mp, S, SSf
import numpy as np


class MP(Generic[S]):
    
    def __init__(self, tr: SSf) -> None:
        if verify_mp(tr):
            self.all_state_list = get_all_states(tr)
            self.transition_matrix: np.ndarray = get_transition_matrix(tr)
        else:
            raise ValueError
    
    # get all states as a set
    def get_states(self) -> Set:
        return set(self.all_state_list)
    
    # get the transition probability matrix as a numpy.array
    def get_tran_mat(self) -> np.ndarray:
        return self.transition_matrix
    
    # compute the stationary distribution 
    def stationary_distribution(self) -> Mapping[S, float]:
        eig_vals, eig_vecs = np.linalg.eig(self.transition_matrix.T)
        stat = np.array(
            eig_vecs[:, np.where(np.abs(eig_vals - 1.) < 1e-8)[0][0]].flat
        ).astype(float)
        norm_stat = stat / sum(stat)
        return {s: norm_stat[i] for i, s in enumerate(self.all_state_list)}
        