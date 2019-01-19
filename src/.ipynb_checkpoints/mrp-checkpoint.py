import numpy as np
from src.mp import MP
from src.my_funcs import SSf, S
from typing import Sequence

class MRP(MP):
    
    # additional function inputs: reward and discount
    def __init__(self, tr:SSf, state_reward: dict, gamma: float) -> None:
        super().__init__(tr)
        if gamma <0 or gamma >1 or set(state_reward.keys()) != set(self.all_state_list):
            raise ValueError
        else:
            self.gamma: float = gamma
            self.terminal_states = self.get_sink_states()
            self.nt_states_list: Sequence[S] = self.get_nt_states_list()
            self.trans_matrix: np.ndarray = self.get_trans_matrix()
            self.reward = np.array([state_reward[s] for s in self.nt_states_list])

    # get non-terminal states
    def get_nt_states_list(self) -> Sequence[S]:
        return [s for s in self.all_state_list
                if s not in self.terminal_states]
    
    # get transition matrix without non-terminal states
    def get_trans_matrix(self) -> np.ndarray:
        
        n = len(self.nt_states_list)
        m = np.zeros((n, n))
        for i in range(n):
            for s, d in self.process[self.nt_states_list[i]].items():
                if s in self.nt_states_list:
                    m[i, self.nt_states_list.index(s)] = d
        return m
    
    # obtain value function of the MRP proecess
    def valueFun(self) -> float:
        return np.linalg.inv(np.identity(len(self.nt_states_list)) - \
                self.gamma*self.trans_matrix).dot(self.reward)
        
        