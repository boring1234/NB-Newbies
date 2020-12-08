# Rllib docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
import sys
import time
import json
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from numpy.random import rand

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo


class DiamondCollector(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.size = 50
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 5
        self.max_episode_steps = 100
        self.log_frequency = 10

        self.max_global_steps = 10000
        self.replay_buffer_size = 10000
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.1
        self.batch_size = 128
        self.gamma = 0.9
        self.target_update = 100
        self.learning_rate = 1e-4
        self.start_training = 500
        self.learn_frequency = 1

        self.action_dict = {
            0: 'move 1',  # 向前一步
            1: 'move -1',  # 向后一步
            2: 'attack 1',  # 攻击
            # 3: 'attack 0'  # 停止攻击
        }

        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(0, 2, shape=(np.prod([1, 1, 2*self.obs_size+1]), ), dtype=np.int32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # Creeper Surviver Parameters
        self.damage_to_score_rate = 0.2
        self.CREEPER = 2
        self.GUNPOWDER = 1
        self.NOTHING = 0
        self.R = 0.25

        self.obs = None
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        print("reset")
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > self.log_frequency and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs, allow_attack_action = self.get_observation(world_state)

        return self.obs.flatten()

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """
        print("step")

        # Get Action
        command = self.action_dict[action]
        allow_attack_action = self.obs[0][0][6] == self.CREEPER
        # allow_break_action = self.obs[1, int(self.obs_size/2)-1, int(self.obs_size/2)] == 1
        if command != 'attack 1':
            self.agent_host.sendCommand(command)
            time.sleep(.1)
            self.episode_step += 1
        elif command == 'attack 1' and allow_attack_action:
            self.agent_host.sendCommand(command)
            self.agent_host.sendCommand('attack 0')

        # Get Done
        done = False
        # if self.episode_step >= self.max_episode_steps or \
        #         (self.obs[0, int(self.obs_size/2)-1, int(self.obs_size/2)] == 1 and \
        #         self.obs[1, int(self.obs_size/2)-1, int(self.obs_size/2)] == 0 and \
        #         command == 'move 1'):
        #     done = True
        #     time.sleep(2)  
        if self.episode_step >= self.max_episode_steps or\
            ((self.obs[0][0][3]==2 or self.obs[0][0][4]==2 or self.obs[0][0][5]==2 or self.obs[0][0][6]==2 or self.obs[0][0][7]==2 or self.obs[0][0][8]==2) and (command == 'move 1' or command == 'move -1')):
            print('done is true now!!!')
            done = True
            time.sleep(2)
        
        for i in range(3, 11):
                if(self.obs[0][0][i] == 2):
                    print("while 2: obs = {}".format(i))
        
        print(command)

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, allow_attack_action = self.get_observation(world_state) 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
            print("reward: {}".format(reward))

        for i in range(3, 8):
                if (self.obs[0][0][i] == 2):
                    reward -= 1
        
        for i in range(0, 3):
                if (self.obs[0][0][i] == 2):
                    reward += 1

        self.episode_return += reward

        return self.obs.flatten(), reward, done, dict()

    def get_mission_xml(self):
        print("get_xml")
        return f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                    <About>
                        <Summary>Diamond Collector</Summary>
                    </About>
                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>12000</StartTime>
                                <AllowPassageOfTime>true</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>
                                <DrawCuboid x1="50" x2="-50" y1="1" y2="1" z1="50" z2="-50" type="stone"/>
                                <DrawCuboid x1="50" x2="-50" y1="2" y2="2" z1="50" z2="-50" type="air"/>
                                <DrawCuboid x1="50" x2="-50" y1="3" y2="3" z1="50" z2="-50" type="air"/>
                                <DrawCuboid x1="1" x2="1" y1="2" y2="3" z1="-50" z2="50" type="obsidian"/>
                                <DrawCuboid x1="0" x2="0" y1="1" y2="1" z1="-50" z2="50" type="obsidian"/>
                                <DrawCuboid x1="-1" x2="-1" y1="2" y2="3" z1="-50" z2="50" type="obsidian"/>
                                <DrawEntity x="0.5" y="2" z="10" type="Creeper" yaw="0"/>
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>
                    <AgentSection mode="Survival">
                        <Name>CS175CreeperSurviver</Name>
                        <AgentStart>
                            <Placement x="0.5" y="2" z="0.5" pitch="45" yaw="0"/>
                            <Inventory>
                                <InventoryItem slot="0" type="diamond_sword"/>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <DiscreteMovementCommands/>
                            <ObservationFromFullStats/>
                            <ObservationFromNearbyEntities>
                                <Range name="entities" xrange="{self.obs_size}" yrange="1" zrange="{self.obs_size}" />
                            </ObservationFromNearbyEntities>
                            <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="0" y="-1" z="-{self.obs_size}"/>
                                    <max x="0" y="-1" z="{self.obs_size}"/>
                                </Grid>
                            </ObservationFromGrid>
                            <ObservationFromRay/>
                            <RewardForDamagingEntity>
                                <Mob type="Creeper" reward="1"/>
                            </RewardForDamagingEntity>
                            <AgentQuitFromReachingCommandQuota total="{self.max_episode_steps}" />
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''

    def init_malmo(self):
        print('init_malmo')
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        print(1)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)
        print(2)
        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
        print(3)

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'DiamondCollector' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)
        print(4)
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        print("get_observation")
        """
        Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array>
        """
        obs = np.zeros((1, 1, 2*self.obs_size+1))
        allow_attack_action = False

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)

                # Get observation
                grid = observations['floorAll']
                grid_binary = [self.GUNPOWDER if x == 'gunpowder' else self.NOTHING for x in grid]
                obs = np.reshape(grid_binary, (1, 1, 2*self.obs_size+1)) # (1,1,11) observation size

                agent_Z = [ent['z'] for ent in observations['entities'] if ent['name']=='CS175CreeperSurviver'][0]

                for ent in observations['entities']:
                    if ent['name'] == 'Creeper':
                        obs[0,0,round(ent['z']-agent_Z)+5] = self.CREEPER  ## 
                for i in range(0, 8):
                    if(obs[0][0][i] == 2):
                        print("get_obs: obs = {}".format(i))

                allow_attack_action = obs[0][0][6] == self.CREEPER

                # Rotate observation with orientation of agent
                yaw = observations['Yaw']
                if yaw == 270:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw == 0:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw == 90:
                    obs = np.rot90(obs, k=3, axes=(1, 2))
                
                break

        return obs, allow_attack_action

    def log_returns(self):
        print('log_returns')
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns, box, mode='same')
        plt.clf()
        plt.plot(self.steps, returns_smooth)
        plt.title('Diamond Collector')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps, self.returns):
                f.write("{}\t{}\n".format(step, value)) 


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=DiamondCollector, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
