import numpy as np
from src.mrp import MRP
from src.my_funcs import S, A, SASf, verify_mdp, get_all_states, SAf 
from typing import Mapping, Set, Sequence, List, Tuple


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
    
    
    def policy_iteration(self, init_pol: Mapping[S,Mapping[A, float]], max_iter: int = 200) \
                     -> Tuple[Mapping[S,Mapping[A, float]], Mapping[S, float]]:
        pol = None
        pol_new = init_pol
        n_iter = 0
        while pol_new != pol and n_iter < max_iter:
            pol = pol_new
            val = self.policy_evaluation(pol) # Sf
            for s in self.terminal_states: # add terminal states
                val[s] = [s for s in self.state_reward[s].values()][0]
            pol_new = {} # SAf
            for s in self.nt_states_list:
                R_sa = self.state_reward[s] # Af
                prob = self.process[s] # ASf
                val_a = {} # Af
                for action in prob.keys():
                    val_a[action] = self.gamma*sum([prob[action][state]*val[state] \
                                        for state in prob[action].keys()])
                q_sa = {a: R_sa[a] + val_a[a] for a in prob.keys()}       
                pol_new[s] = { max(q_sa.items(), key=lambda l: l[1])[0]: 1}
            # terminal state policy
            for state in self.terminal_states:
                pol_new[state] = init_pol[state]
            n_iter += 1
        
        if pol_new == pol:
            print("Number of iterations: {}.".format(n_iter))
        else:
            print("Not converging in {} iterations.".format(n_iter))
        
        return pol, val
    
                
    def value_iteration(self, eps: float = 1e-8, max_iter: int = 200) \
                     -> Tuple[Mapping[S,Mapping[A, float]], Mapping[S, float]]:
        val = {s: 0 for s in self.all_state_list}
        val_new = {}
        diff = 1000
        n_iter = 0
        pol = {}
        while diff > eps and n_iter < max_iter:
            for s in self.all_state_list:
                q_sa = {}
                for a in self.process[s].keys():
                    prob = self.process[s][a]
                    q_sa[a] = self.state_reward[s][a] + self.gamma*\
                        sum([prob[state]*val[state] for state in prob.keys()])
                find_max = max(q_sa.items(), key=lambda l: l[1])
                val_new[s] = find_max[1]
                pol[s] = { find_max[0]: 1}
            diff = max([np.abs(val_new[s] - val[s]) for s in self.all_state_list])
            val = val_new.copy()
            n_iter += 1
        
        if diff <= eps:
            print("Number of iterations: {}.".format(n_iter))
        else:
            print("Not converging in {} iterations.".format(n_iter))
            
        return pol, val
        