import numpy as np
from src.mrp import MRP
from src.my_funcs import S, A, SASf, verify_mdp, get_all_states, SAf
from typing import Mapping, Set, Sequence, List


class MDP(MRP):
    
    # additional function inputs: reward and discount
    def __init__(self, tr:SASf, state_reward: SAf, gamma: float) -> None:
        if verify_mdp(tr):
            self.process: SASf = tr
            self.all_state_list: List[S] = get_all_states(tr)
            self.transition_matrix: Mapping[S,np.ndarray] = self.get_trans_matrix()
            if gamma <0 or gamma >1 or set(state_reward.keys()) != set(self.all_state_list):
                raise ValueError
            else:
                self.gamma = gamma
                self.terminal_states: Set[S] = self.get_sink_states()
                self.nt_states_list: Sequence[S] = self.get_nt_states_list()
                self.reward = np.array([state_reward[s] for s in self.nt_states_list])
                self.state_reward = state_reward
        else:
            raise ValueError
    
    
    # get the terminal state
    def get_sink_states(self) -> Set[S]:
        return {k for k, v in self.process.items()  \
         if all(len(v1) == 1 and k in v1.keys() for k1, v1 in v.items())}
        
    
    # The transition matrix for each state is a ndarray of actions and states
    def get_trans_matrix(self) -> Mapping[S,np.ndarray]:
        mat = {}
        m = len(self.all_state_list)
        for state in self.all_state_list:
            dict_s = self.process[state]
            n = len(dict_s.keys())
            ndarr = np.zeros((n,m))
            for i, (s, v) in enumerate(dict_s.items()):
                for j, st in enumerate(v.keys()):
                    ndarr[i,j] = v[st]
            mat[state] = ndarr
        
        return mat
    
    
    # given a input policy return a MRP
    def get_mrp(self, pol: Mapping[S,Mapping[A, float]]) -> MRP:
        tr = {}
        reward = {}
        for state in self.all_state_list:
            pi = pol[state]
            pro = self.process[state]
            re = self.state_reward[state]
            tr[state] = {}
            reward[state] = 0
            for action, p in pi.items():
                reward[state] += re[action]*p
                for s in pro[action].keys():
                    if s in tr[state].keys():
                        tr[state][s] += pro[action][s]*p
                    else:
                        tr[state][s] = pro[action][s]*p
                    
        return MRP(tr, reward, self.gamma)
    
    
    def policy_evaluation(self, pol: Mapping[S,Mapping[A, float]]) -> Mapping[S, float]:
        return {state: self.get_mrp(pol).valueFun()[i] \
                 for i, state in enumerate(self.nt_states_list)}
                    
    
        
 
