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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Hyperparameters
SIZE = 50
REWARD_DENSITY = .1
PENALTY_DENSITY = .02
OBS_SIZE = 5
MAX_EPISODE_STEPS = 100
MAX_GLOBAL_STEPS = 10000
REPLAY_BUFFER_SIZE = 10000
EPSILON_DECAY = .999
MIN_EPSILON = .1
BATCH_SIZE = 128
GAMMA = .9
TARGET_UPDATE = 100
LEARNING_RATE = 1e-4
START_TRAINING = 500
LEARN_FREQUENCY = 1
ACTION_DICT = {
    0: 'move 1',
    1: 'move -1',
    2: 'turn 1',
    3: 'turn -1',
    4: 'attack 1'
}

CREEPER = 2
GUNPOWDER = 1
NOTHING = 0
TOTAL_CREEPER = 0

class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(np.prod(obs_size), hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, action_size)) 
        
    def forward(self, obs):

        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        return self.net(obs_flat)


def GetMissionXML():
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
                            
                            <DrawCuboid x1="-25" x2="-25" y1="2" y2="3" z1="-50" z2="50" type="obsidian"/>
                            <DrawCuboid x1="-24" x2="24" y1="1" y2="1" z1="-50" z2="50" type="obsidian"/>
                            <DrawCuboid x1="25" x2="25" y1="2" y2="3" z1="-50" z2="50" type="obsidian"/>
                            
                            <DrawCuboid x1="-24" x2="24" y1="2" y2="3" z1="50" z2="50" type="obsidian"/>
                            <DrawCuboid x1="-24" x2="24" y1="2" y2="3" z1="-50" z2="-50" type="obsidian"/>
                            
                            <DrawEntity x="0.5" y="2" z="10" type="Creeper" yaw="0"/>

                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>
                <AgentSection mode="Survival">
                    <Name>CS175CreeperSurviver</Name>
                    <AgentStart>
                        <Placement x="0.5" y="2" z="0.5" pitch="30" yaw="0"/>
                        <Inventory>
                            <InventoryItem slot="0" type="diamond_sword"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <ContinuousMovementCommands turnSpeedDegs="180"/>
                        <ObservationFromFullStats/>
                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="{OBS_SIZE}" yrange="1" zrange="{OBS_SIZE}" />
                        </ObservationFromNearbyEntities>
                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                <min x="-{OBS_SIZE}" y="-1" z="-{OBS_SIZE}"/>
                                <max x="{OBS_SIZE}" y="-1" z="{OBS_SIZE}"/>
                            </Grid>
                        </ObservationFromGrid>
                        <ObservationFromRay/>
                        <RewardForDamagingEntity>
                            <Mob type="Creeper" reward="5"/>
                        </RewardForDamagingEntity>
                        <AgentQuitFromReachingCommandQuota total="{MAX_EPISODE_STEPS}" />
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


def get_action(obs, q_network, epsilon, allow_attack_action):


    with torch.no_grad():
        obs_torch = torch.tensor(obs.copy(), dtype=torch.float).unsqueeze(0)
        action_values = q_network(obs_torch)

        if not allow_attack_action:
            action_values[0, 4] = -float('inf')

        action_idx = torch.argmax(action_values).item()

        
    if allow_attack_action:
        return action_idx if epsilon < 0.27 else random.randint(0, 4)
    else:
        return action_idx if epsilon < 0.27 else random.randint(0, 3)


def init_malmo(agent_host):

    my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)

    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_clients, my_mission_record, 0, "DiamondCollector" )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    return agent_host


def get_observation(world_state):

    obs = np.zeros((1, 2 * OBS_SIZE + 1, 2 * OBS_SIZE + 1))
    # print(obs)
    CreeperInRange = False
    life = 20

    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            # print(json.dumps(observations, indent=4, sort_keys=True))
            grid = observations['floorAll']
            grid_binary = [GUNPOWDER if x == 'gunpowder' else NOTHING for x in grid]
            obs = np.reshape(grid_binary, (1, 2 * OBS_SIZE + 1, 2 * OBS_SIZE + 1))
            # print(obs)

            agent_Z = [ent['z'] for ent in observations['entities'] if ent['name']=='CS175CreeperSurviver'][0]
            for ent in observations['entities']:
                if ent['name'] == 'Creeper':
                    obs[0,round(ent['z']-agent_Z)+5,round(ent['z']-agent_Z)+5] = CREEPER
            print (obs)
            if (observations['LineOfSight']['type'] == 'Creeper' and observations['LineOfSight']['inRange'] == True):
                CreeperInRange=True


            # life =  observations['Life']

            yaw = observations['Yaw']
            if yaw == 270:
                obs = np.rot90(obs, k=1, axes=(1, 2))
            elif yaw == 0:
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif yaw == 90:
                obs = np.rot90(obs, k=3, axes=(1, 2))
            
            break

    return obs, CreeperInRange, life


def prepare_batch(replay_buffer):
    """
    Randomly sample batch from replay buffer and prepare tensors

    Args:
        replay_buffer (list): obs, action, next_obs, reward, done tuples

    Returns:
        obs (tensor): float tensor of size (BATCH_SIZE x obs_size
        action (tensor): long tensor of size (BATCH_SIZE)
        next_obs (tensor): float tensor of size (BATCH_SIZE x obs_size)
        reward (tensor): float tensor of size (BATCH_SIZE)
        done (tensor): float tensor of size (BATCH_SIZE)
    """
    batch_data = random.sample(replay_buffer, BATCH_SIZE)
    obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
    action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
    next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
    reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
    done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)
    
    return obs, action, next_obs, reward, done
  

def learn(batch, optim, q_network, target_network):
    """
    Update Q-Network according to DQN Loss function

    Args:
        batch (tuple): tuple of obs, action, next_obs, reward, and done tensors
        optim (Adam): Q-Network optimizer
        q_network (QNetwork): Q-Network
        target_network (QNetwork): Target Q-Network
    """
    obs, action, next_obs, reward, done = batch

    optim.zero_grad()
    values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
    target = torch.max(target_network(next_obs), 1)[0]
    target = reward + GAMMA * target * (1 - done)
    loss = torch.mean((target - values) ** 2)
    loss.backward()
    optim.step()

    return loss.item()


def log_returns(steps, returns):
    """
    Log the current returns as a graph and text file

    Args:
        steps (list): list of global steps after each episode
        returns (list): list of total return of each episode
    """
    box = np.ones(10) / 10
    returns_smooth = np.convolve(returns, box, mode='same')
    plt.clf()
    plt.plot(steps, returns_smooth)
    plt.title('Diamond Collector')
    plt.ylabel('Return')
    plt.xlabel('Steps')
    plt.savefig('returns.png')

    with open('returns.txt', 'w') as f:
        for value in returns:
            f.write("{}\n".format(value)) 


def train(agent_host):
    """
    Main loop for the DQN learning algorithm

    Args:
        agent_host (MalmoPython.AgentHost)
    """
    # Init networks
    q_network = QNetwork((1, 2 * OBS_SIZE + 1, 2 * OBS_SIZE + 1), len(ACTION_DICT))
    target_network = QNetwork((1, 2 * OBS_SIZE + 1, 2 * OBS_SIZE + 1), len(ACTION_DICT))
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # Init replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Init vars
    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []

    # Begin main loop
    loop = tqdm(total=MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < MAX_GLOBAL_STEPS:
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False
        # Setup Malmo
        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:",error.text)
        obs, CreeperInRange, life = get_observation(world_state)
        # Run episode
        creeper_dict = {}
        while world_state.is_mission_running:
            # Get action
            

            allow_break_action = CreeperInRange==True
            action_idx = get_action(obs, q_network, epsilon, allow_break_action)
            command = ACTION_DICT[action_idx]
            # Take step
            agent_host.sendCommand(command)
            # If your agent isn't registering reward you may need to increase this
            time.sleep(.1)

            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            if episode_step >= MAX_EPISODE_STEPS or life == 0:
                done = True
                time.sleep(2)  

            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs, next_CreeperInRange, next_life = get_observation(world_state) 

            # Get reward
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()

            if life < 10 and life > 0 :
                reward -= 3
            
            if life > 10 and life < 15 :
                reward -= 1.5

            if life > 15 and life < 20 :
                reward += 10

            if life == 0:
                print("i am dead")
                reward -= 20

            episode_return += reward
            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs, CreeperInRange, life = next_obs, next_CreeperInRange, next_life

            # Learn
            global_step += 1
            if global_step > START_TRAINING and global_step % LEARN_FREQUENCY == 0:
                batch = prepare_batch(replay_buffer)
                loss = learn(batch, optim, q_network, target_network)
                episode_loss += loss

                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY

                if global_step % TARGET_UPDATE == 0:
                    target_network.load_state_dict(q_network.state_dict())

        num_episode += 1
        returns.append(episode_return)
        steps.append(global_step)
        avg_return = sum(returns[-min(len(returns), 10):]) / min(len(returns), 10)
        loop.update(episode_step)
        loop.set_description('Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.2f}'.format(
            num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode > 0 and num_episode % 10 == 0:
            log_returns(steps, returns)
            print()


if __name__ == '__main__':
    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    train(agent_host)
