from src.mdp_for_rl_tabular import MDPRLTabular
from src.mdp import MDP
from src.my_funcs import SASTff, mdp_refined_split_info,get_state_reward_gen_dict, get_actions_for_states

class MDPRefined(MDP):
    
    def __init__(self, inf: SASTff, gamma: float) -> None:
        tr, rr, state_reward = mdp_refined_split_info(inf)
        super().__init__(tr, state_reward, gamma)
        self.state_action_dict = get_actions_for_states(inf)
        self.rewards_refined = rr
    
    def get_mdp_rep_for_rl_tabular(self) -> MDPRLTabular:
        return MDPRLTabular(
            state_action_dict=self.state_action_dict,
            terminal_states=self.terminal_states,
            state_reward_gen_dict=get_state_reward_gen_dict(
                self.process,
                self.rewards_refined
            ),
            gamma=self.gamma
        )     
        