import json
import os

class BenchmarkHandler:

    def __init__(self, data_path: str="/data_arl_bench", environment: str="Pong-v0", search_space: str="PPO", 
                       static: bool=True, return_names=None):

        self.data_path = data_path
        
        self.environment = environment
        self.search_space = search_space
        self.return_names = return_names
        self.static = static
        self.search_space_structure =  {"static": {"PPO": {"lr": [-6, -4, -4, -3, -2,-1], 
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99,1.0],
                                                            "clip": [0.2, 0.3, 0.4]},
                                                    "A2C": {"lr": [-6, -5, -4, -3, -2, -1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0]},
                                                    "DDPG": {"lr": [-6, -5, -4, -3, -2, -1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0]}},
                                        "dynamic": {"PPO": {"lr": [-5, -4, -3],
                                                            "gamma": [0.95, 0.98, 0.99]}}
                                        }
        self.environment_list = {"atari": ["Pong-v0", "Alien-v0", "BankHeist-v0", "BeamRider-v0", "Breakout-v0", "Enduro-v0", "Phoenix-v0",
                                "Seaquest-v0", "SpaceInvadores-v0", "Riverraid-v0", "Tennis-v0", "Skiing-v0", "Boxing-v0", 
                                "Bowling-v0", "Asteroids-v0"], 
                                 "mujoco" : ["Ant-v2", "Hopper-v2", "Humanoid-v2"],
                                 "control" : ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v0"]}
        self.seeds_list = [0,1,2]
        self.env_types = ["atari", "mujoco", "control"]

        if self.return_names == None:
            self.return_names = ["returns_eval", "std_returns_eval", "timestamps_eval", "timesteps_eval", "timesteps_train", "timestamps_train", "returns_train"]


    def set_env_and_space(self, search_space: str, environment : str):

        self.search_space = search_space
        self.environment = environment


    def __build_return_dict(self, data, budget, train_timesteps_index):

        print(self.return_names, data)
        max_budget_allowed = len(data["timesteps_eval"])
        assert budget < max_budget_allowed, f"Budget should be lower than {max_budget_allowed}"

        if self.return_names == None:
                    return {
                        "returns_eval": data["returns_eval"][:budget],
                        "std_returns_eval": data["std_returns_eval"][:budget],
                        "timestamps_eval": data["timestamps_eval"][:budget],
                        "timesteps_eval": data["timesteps_eval"][:budget],

                        "timesteps_train": data["timesteps_train"][:train_timesteps_index],
                        "timestamps_train": data["timestamps_train"][:train_timesteps_index],
                        "returns_train": data["returns_train"][:train_timesteps_index]
                    }

        else:
            return_dict = {}
            for name in self.return_names:
                if "eval" in name:
                    return_dict[name] = data[name][:budget]
                elif "train" in name:
                    return_dict[name] = data[name][:train_timesteps_index]
            return return_dict


    def get_environments (self, env_type: bool="atari"):

        return self.environment_list[env_type]

    def get_search_spaces_names(self, static: bool=True):

        if static:
            return list(self.search_space_structure["static"].keys())
        else:
            return list(self.search_space_structure["dynamic"].keys())



    def get_search_space(self, search_space, static: bool=True):

        if static:
            return self.search_space_structure["static"][search_space]
        else:
            return self.search_space_structure["dynamic"][search_space]

    def get_metrics(self, config: dict, search_space: str="", environment: str="" , seed: int=0, budget: int=199, static: bool=True):

        if search_space == "":
            assert self.search_space != None, "Please set the search space"
            search_space = self.search_space
            static = self.static
        
        if environment == "":
            assert self.environment != None, "Please set the environment"
            environment = self.environment


        if static:
            lr = config.get("lr")
            gamma = config.get("gamma")
            if search_space == "DQN":
                epsilon = config.get("epsilon")
                with open(os.path.join(self.data_path, 'data_arl_bench', search_space, environment,
                                    '%s_%s_random_lr_%s_gamma_%s_gamma_%s_seed%s_eval.json'%(environment, search_space,
                                                                                        lr, gamma, epsilon, seed))) as f:
                    data = json.load(f)
                    train_timesteps_index = data["timesteps_train"].index(data["timesteps_eval"][budget])
                return self.__build_return_dict(data, budget, train_timesteps_index)

            elif search_space == "PPO":
                clip = config.get("clip")
                with open(os.path.join(self.data_path, 'data_arl_bench', search_space, environment,
                                    '%s_%s_random_lr_%s_gamma_%s_clip_%s_seed%s_eval.json'%(environment, search_space,
                                                                                        lr, gamma, clip, seed))) as f:
                    data = json.load(f)
                    train_timesteps_index = data["timesteps_train"].index(data["timesteps_eval"][budget])

                return self.__build_return_dict(data, budget, train_timesteps_index)
            
            elif search_space == "A2C":
                with open(os.path.join(self.data_path, 'data_arl_bench', search_space, environment,
                                    '%s_%s_random_lr_%s_gamma_%s_seed%s_eval.json'%(environment, search_space,
                                                                                        lr, gamma, seed))) as f:
                    data = json.load(f)
                    train_timesteps_index = data["timesteps_train"].index(data["timesteps_eval"][budget])
                return self.__build_return_dict(data, budget, train_timesteps_index)
            
        else:
            lrs = config.get("lr")
            if len(lrs) == 1:
                lrs = lrs * 3
            elif len(lrs) == 2:
                lrs = lrs.append[lrs[1]]

            gammas = config.get("gamma")
            if len(gammas) == 1:
                gammas = gammas * 3
            elif len(gammas) == 2:
                gammas = gammas.append[gammas[1]]
            if search_space == "DQN":
                with open(os.path.join(self.data_path, 'data_arl_bench', search_space, environment,
                                    '%s_%s_random_lr_%s%s%s_gamma_%s%s%s_seed%s_eval.json'%(environment, search_space,
                                                                                            lrs[0], lrs[1], lrs[2],
                                                                                            gammas[0], gammas[1],
                                                                                            gammas[2], seed))) as f:
                    data = json.load(f)
                    train_timesteps_index = data["timesteps_train"].index(data["timesteps_eval"][budget-1])
                return self.__build_return_dict(data, budget, train_timesteps_index)






