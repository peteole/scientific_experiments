
from dataclasses import dataclass
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces
import torch
from torch import Tensor
from typing import Any, Type, TypedDict, List, Tuple
import  torch.distributions
from tensordict import TensorDict


class TrafficLightsPhysicsConfig(TypedDict):
    """Traffic lights physics configuration"""
    
    mass: float # Mass of the car in kg
    power: float # Power of the car in Watts
    C_d: float # Drag coefficient of the car
    A: float # Frontal area of the car in m^2
    rho: float # Density of air in kg/m^3
    brake_deceleration: float # Maximum deceleration when braking in m/s^2
    max_acceleration: float # Maximum acceleration in m/s^2
    max_speed: float # Maximum allowed speed in m/s
    start_position: float # Starting position of the car in m
    start_speed: float # Starting speed of the car in m/s
    dt: float # Time step in seconds
    initial_ttg_distribution: torch.distributions.Distribution # Distribution of initial time to green
    red_crossing_penalty: float # Penalty for crossing the red light in USD
    costs_per_second: float # Costs per second time loss in USD
    costs_per_joule: float # Costs per Joule energy loss in USD


class TrafficLightsEnv(VecEnv):
    """
    observation space: (num_envs, num_obs). Observations: [speed, position, red(0)/green(1), time]
    state space: (num_envs, state_param_count). States: [speed, position, time_green, time]
    """

    def __init__(self,  num_envs: int, device: torch.device, physics_config: TrafficLightsPhysicsConfig):
        self.physics_config = physics_config
        self.num_envs = num_envs
        self.device = device
        self.action_space =spaces.Box(low=-physics_config['brake_deceleration'],high=physics_config['max_acceleration'])
        self.observation_space = spaces.Box(
            low=np.array((0.0, physics_config['start_position'], 0.0, 0.0)),
            high=np.array((physics_config['max_speed'], 1000.0, 1.0, physics_config['initial_ttg_distribution'].icdf(0.99)+10)),
        )
        self.states=self.sample_new_states(num_envs)
        self.actions: torch.Tensor = torch.zeros((num_envs,), device=device)
        self.base_energy_per_meter:float=self.compute_power(self.physics_config['max_speed'],0)/self.physics_config['max_speed']

    
    def step_async(self, actions: np.ndarray) -> None:
        self.actions = torch.tensor(actions, device=self.device)

    def sample_new_states(self, num_states: int) -> TensorDict:
        ttg=self.physics_config['initial_ttg_distribution'].sample((num_states,)).to(self.device)
        positions=torch.ones((num_states,),device=self.device)*self.physics_config['start_position']
        speeds=torch.ones((num_states,),device=self.device)*self.physics_config['start_speed']
        start_times=torch.zeros((num_states,),device=self.device)
        return TensorDict({'ttgs':ttg,'positions':positions,'speeds':speeds,'times':start_times},batch_size=num_states,device=self.device)

    
    def compute_max_acceleration(self, current_speeds: torch.Tensor) -> torch.Tensor:
        return torch.min(
            torch.ones(self.num_envs,device=self.device)*self.physics_config['max_acceleration'],
            self.physics_config['power']/(current_speeds+1e-6)/self.physics_config['mass'])
    def compute_power(self, current_speeds: Tensor, accelerations: Tensor) -> Tensor:
        return accelerations*current_speeds*self.physics_config['mass'] \
                + 0.5*self.physics_config['C_d']*self.physics_config['A']*self.physics_config['rho']*current_speeds**3
    
    def get_observations(self) -> Tensor:
        return torch.stack((self.states['speeds'],self.states['positions'],self.states['ttgs']>self.states['times'],self.states['times']),dim=1).cpu().numpy()
    def step_wait(self) -> tuple:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        dt=self.physics_config['dt']
        current_states=self.states
        current_speeds=current_states['speeds']
        max_accelerations=self.compute_max_acceleration(current_speeds)
        accelerations=torch.clamp(self.actions.flatten(),-torch.ones(self.num_envs, device=self.device)*self.physics_config['brake_deceleration'],max_accelerations)
        dx=self.states['speeds']*dt+0.5*accelerations*dt**2
        next_positions=self.states['positions']+dx
        next_speeds=self.states['speeds']+accelerations*dt
        next_times=self.states['times']+dt
        rewards=torch.zeros((self.num_envs,),device=self.device)

        next_lights_red=next_times<self.states['ttgs']
        dones= ~next_lights_red & (next_speeds>=self.physics_config['max_speed']*0.99)

        next_states=TensorDict({'ttgs':self.states['ttgs'],'positions':next_positions,'speeds':next_speeds,'times':next_times},batch_size=self.num_envs,device=self.device)

        # penalize crossing the red light if this happens this time step
        crossing_red=next_lights_red & (next_positions>=0) & (self.states['positions']<0)
        rewards[crossing_red]=self.physics_config['red_crossing_penalty']

        # penalize slow speeds
        time_loss=dt*(1-next_speeds/self.physics_config['max_speed'])
        rewards -= self.physics_config['costs_per_second']*time_loss
        energy_loss=self.compute_power(next_speeds,accelerations)*dt-self.base_energy_per_meter*dx
        rewards -= self.physics_config['costs_per_joule']*energy_loss
        overspeed_penalty=(next_speeds-self.physics_config['max_speed']).exp()*dt
        rewards -= self.physics_config['costs_per_second']*overspeed_penalty

        next_states[dones]=self.sample_new_states(dones.sum().item())

        self.states=next_states

        return self.get_observations(), rewards.cpu().numpy(), dones.int().cpu().numpy(), [{}]*self.num_envs

    def reset(self) -> Tensor:
        self.states=self.sample_new_states(self.num_envs)
        return self.get_observations()

    def close(self) -> None:
        del self.states
        del self.actions
    
    def seed(self, seed: int) -> None:
        torch.manual_seed(seed)
    
    def env_is_wrapped(self, wrapper_class, indices = None) -> List[bool]:
        return [False]*self.num_envs
    
    def get_attr(self, attr_name: str, indices = None) -> List[Any]:
        print('get_attr',attr_name,"not implemented")
        return None

    def set_attr(self, attr_name: str, value: Any, indices = None) -> None:
        print('set_attr',attr_name,"not implemented")
    
    def env_method(self, method_name: str, *method_args, indices = None, **method_kwargs) -> List[Any]:
        print('env_method',method_name,"not implemented")
        return None
    

