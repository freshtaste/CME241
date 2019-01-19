import numpy as np
from typing import Tuple
from src.my_funcs import SSf, get_all_states, get_transition_matrix
from src.mp import MP
from src.mrp import MRP

class MRP2(MP):
    # implement the second definition of MRP
    
    # additional function inputs: reward and discount
    def __init__(self, tr: SSf, reward: SSf, gamma: float) -> None:
        super().__init__(tr)
        if gamma <0 or gamma >1 or set(get_all_states(reward)) != set(self.all_state_list):
            raise ValueError
        else:
            self.reward_mat = get_transition_matrix(reward)
            self.gamma = gamma
    
    # convert from the second definiton to first definition
    def convert(self) -> Tuple[SSf, dict, float]:
        prob = self.transition_matrix
        val = self.reward_mat
        reward_list = np.diag(prob.dot(val.T)).tolist()
        return self.process, {s:reward_list[i] for i, s in enumerate(self.all_state_list)}, self.gamma
    
    def valueFun(self) -> float:
        obj = MRP(self.convert)
        return obj.valueFun()