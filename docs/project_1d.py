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

from numpy.random import rand


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
    0: 'move 1',  # 向前一步
    1: 'move -1',  # 向后一步
    2: 'attack 1',  # 攻击
    3: 'attack 0' # 停止攻击
}
### https://github.com/microsoft/malmo/blob/master/Schemas/MissionHandlers.xsd 这里有所有玩家动作

##这里加了伤害转换成分数的比例系数
DAMAGE_TO_SCORE_RATE = 0.2  


# Q-Value Network
class QNetwork(nn.Module):
    #------------------------------------
    #
    #   TODO: Modify network architecture
    #
    #-------------------------------------

    def __init__(self, obs_size, action_size, hidden_size=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(np.prod(obs_size), hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, action_size)) 
        
    def forward(self, obs):
        """
        Estimate q-values given obs
        Args:
            obs (tensor): current obs, size (batch x obs_size)
        Returns:
            q-values (tensor): estimated q-values, size (batch x action_size)
        """
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        return self.net(obs_flat)


def GetMissionXML():
    #------------------------------------
    #
    #   TODO: Spawn diamonds
    #   TODO: Spawn lava
    #   TODO: Add diamond reward
    #   TODO: Add lava negative reward
    #
    #-------------------------------------


    ## 加了 RewardForDamagingEntity，DrawingDecorator, 改了ObservationFromNearbyEntities，ObservationFromGrid
    ## 注意这里如果跑程序时人物不在地图里，请改drawingdecorator里 第4-6行的x1,x2
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
                            <DrawCuboid x1="70" x2="-30" y1="1" y2="1" z1="50" z2="-50" type="stone"/>
                            <DrawCuboid x1="70" x2="-30" y1="2" y2="2" z1="50" z2="-50" type="air"/>
                            <DrawCuboid x1="70" x2="-30" y1="3" y2="3" z1="50" z2="-50" type="air"/>
                            <DrawCuboid x1="21" x2="21" y1="2" y2="3" z1="-50" z2="50" type="obsidian"/>
                            <DrawCuboid x1="20" x2="20" y1="1" y2="1" z1="-50" z2="50" type="obsidian"/>
                            <DrawCuboid x1="19" x2="19" y1="2" y2="3" z1="-50" z2="50" type="obsidian"/>
                            <DrawEntity x="20.5" y="2" z="10" type="Creeper" yaw="0"/>
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>
                <AgentSection mode="Survival">
                    <Name>CS175CreeperSurviver</Name>
                    <AgentStart>
                        <Placement x="20" y="3" z="1.5" pitch="45" yaw="0"/>
                        <Inventory>
                            <InventoryItem slot="0" type="diamond_sword"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <DiscreteMovementCommands/>
                        <ObservationFromFullStats/>
                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="{OBS_SIZE}" yrange="1" zrange="{OBS_SIZE}" />
                        </ObservationFromNearbyEntities>
                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                <min x="0" y="-1" z="-{OBS_SIZE}"/>
                                <max x="0" y="-1" z="{OBS_SIZE}"/>
                            </Grid>
                        </ObservationFromGrid>
                        <RewardForDamagingEntity>
                            <Mob type="Creeper" reward="1"/>
                        </RewardForDamagingEntity>
                        <AgentQuitFromReachingCommandQuota total="{MAX_EPISODE_STEPS}" />
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


def get_action(obs,epsilon):
    """
    Select action according to e-greedy policy
    Args:
        obs (np-array): current observation, size (obs_size)
        q_network (QNetwork): Q-Network
        epsilon (float): probability of choosing a random action
    Returns:
        action (int): chosen action [0, action_size)
    """

    r = rand()
    if r>epsilon:
        # exploitation: choose the argmax
        pass
    else:
        # exploration:
        action_idx = randint(3)
        return action_idx

    #------------------------------------
    #
    #   TODO:这里写通过train完了选的action
    #
    #-------------------------------------
    
    return 0 # 这是我改的。 这可以然玩家直接操控 agent    
    # return action_idx


def init_malmo(agent_host):
    """
    Initialize new malmo mission.
    """
    my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)

    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

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


def get_observation(world_state, life):
    """
    Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
    The agent is in the center square facing up.
    Args
        world_state: <object> current agent world state
    Returns
        observation: <np.array>
    """
    obs = np.zeros((1, 1, 2 * OBS_SIZE+1))  # 这里改了observation size

    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            # First we get the json from the observation API
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            print(observations)#added for debugging
            """
            what the above line prints when the agent looks a round or moves around:
            
            {'DistanceTravelled': 26117, 'TimeAlive': 1767, 'MobsKilled': 0, 'PlayersKilled': 0, 'DamageTaken': 0, 'DamageDealt': 0, 'Life': 20.0, 'Score': 0, 'Food': 20, 'XP': 0, 'IsAlive': True, 'Air': 300, 'Name': 'CS175DiamondCollector', 'XPos': 39.99988746507245, 'YPos': 3.0, 'ZPos': 1.699999988079071, 'Pitch': 33.3, 'Yaw': -286.4995, 'WorldTime': 2135, 'TotalTime': 1779, 'floorAll': ['air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air']}
            life = 20(full)  starting seeing the creeper
            {'DistanceTravelled': 30771, 'TimeAlive': 2778, 'MobsKilled': 0, 'PlayersKilled': 0, 'DamageTaken': 120, 'DamageDealt': 0, 'Life': 20.0, 'Score': 0, 'Food': 20, 'XP': 0, 'IsAlive': True, 'Air': 300, 'Name': 'CS175DiamondCollector', 'XPos': 20.2028200313867, 'YPos': 3.0, 'ZPos': 1.300000011920929, 'Pitch': 45.0, 'Yaw': 0.0, 'WorldTime': 12128, 'TotalTime': 2790, 'entities': [{'yaw': 0.0, 'x': 20.224340942272693, 'y': 3.0, 'z': 1.300000011920929, 'pitch': 45.0, 'id': '642865e8-993c-3421-aad8-35531c528774', 'motionX': 0.004149889454665737, 'motionY': -0.0784000015258789, 'motionZ': 0.0, 'life': 20.0, 'name': 'CS175DiamondCollector'}, {'yaw': -91.40625, 'x': 18.151123046875, 'y': 3.0, 'z': 1.5, 'pitch': 0.0, 'id': 'cd756268-5e4b-4d08-a6ba-b4b8c80f5317', 'motionX': 0.0048006991671951235, 'motionY': -0.057885353419833516, 'motionZ': 0.0, 'life': 20.0, 'name': 'Creeper'}, {'yaw': 270.0, 'x': 17.376220703125, 'y': 3.0, 'z': 1.494384765625, 'pitch': -1.40625, 'id': 'b954c630-8d13-45b1-a341-71a3dade13b7', 'motionX': 0.005310973244769054, 'motionY': -0.06403808123981149, 'motionZ': 0.0, 'life': 20.0, 'name': 'Creeper'}], 'floorAll': ['air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air']}
            keep approaching the creeper
            {'DistanceTravelled': 30811, 'TimeAlive': 2783, 'MobsKilled': 0, 'PlayersKilled': 0, 'DamageTaken': 120, 'DamageDealt': 0, 'Life': 20.0, 'Score': 0, 'Food': 20, 'XP': 0, 'IsAlive': True, 'Air': 300, 'Name': 'CS175DiamondCollector', 'XPos': 19.798259488471064, 'YPos': 3.0, 'ZPos': 1.300000011920929, 'Pitch': 45.0, 'Yaw': 270.0, 'WorldTime': 12133, 'TotalTime': 2795, 'entities': [{'yaw': 270.0, 'x': 21.206472296238157, 'y': 4.6367750149965286, 'z': 1.300000011920929, 'pitch': 45.0, 'id': '642865e8-993c-3421-aad8-35531c528774', 'motionX': 0.4525774181112579, 'motionY': 0.6767145278673175, 'motionZ': 0.0, 'life': 0.0, 'name': 'CS175DiamondCollector'}, {'yaw': 270.0, 'x': 16.835187789036162, 'y': 3.752137637405216, 'z': 1.4904994172267743, 'pitch': -0.46875, 'id': 'b954c630-8d13-45b1-a341-71a3dade13b7', 'motionX': -0.443125, 'motionY': 1.027125, 'motionZ': -0.003125, 'life': 0.0, 'name': 'Creeper'}, {'yaw': 338.90625, 'x': 19.140103745818138, 'y': 4.543785941582918, 'z': 1.561767578125, 'pitch': 0.0, 'id': '543a3424-b5ac-4d41-95fe-a8d542732f00', 'motionX': -0.21056770819644938, 'motionY': 0.03222975375235089, 'motionZ': 0.0, 'quantity': 1, 'name': 'diamond_pickaxe'}, {'yaw': 164.53125, 'x': 17.179746190977863, 'y': 3.2768000057160855, 'z': 1.5916937536250897, 'pitch': 0.0, 'id': '81f2ac87-f77f-4550-9315-3d8eefbd6fb6', 'motionX': -0.09531970371036533, 'motionY': 0.11446400695335877, 'motionZ': 0.04717965183649065, 'quantity': 1, 'name': 'gunpowder'}], 'floorAll': ['air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air']}
            life = 0 after killed by explosion of creeper
            {'DistanceTravelled': 30811, 'TimeAlive': 0, 'MobsKilled': 0, 'PlayersKilled': 0, 'DamageTaken': 390, 'DamageDealt': 0, 'Life': 0.0, 'Score': 0, 'Food': 20, 'XP': 0, 'IsAlive': True, 'Air': 300, 'Name': 'CS175DiamondCollector', 'XPos': 23.37950152675729, 'YPos': 7.354274464662253, 'ZPos': 1.300000011920929, 'Pitch': 45.0, 'Yaw': 180.0, 'WorldTime': 12140, 'TotalTime': 2802, 'entities': [{'yaw': 180.0, 'x': 23.636506371111786, 'y': 7.506239046213171, 'z': 1.300000011920929, 'pitch': 45.0, 'id': '642865e8-993c-3421-aad8-35531c528774', 'motionX': 0.23387441510281104, 'motionY': 0.07052529129251528, 'motionZ': 0.0, 'life': 0.0, 'name': 'CS175DiamondCollector'}, {'yaw': 270.0, 'x': 14.741780598958332, 'y': 8.20263671875, 'z': 1.4853515625, 'pitch': -0.46875, 'id': 'b954c630-8d13-45b1-a341-71a3dade13b7', 'motionX': -0.251625, 'motionY': 0.462375, 'motionZ': 0.0, 'life': 0.0, 'name': 'Creeper'}], 'floorAll': ['air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'obsidian', 'air', 'air', 'air', 'air', 'air']}
            """
            # Get observation
            grid = observations['floorAll']
            grid_binary = [1 if x == 'gunpowder' else 0 for x in grid]  ##这里把gunpowder设成1，不是gunpowder设成0
            # print(grid_binary)
            obs = np.reshape(grid_binary, (1, 1, 2*OBS_SIZE+1))  # 这里改了observation size
            agent_Z = [ent['z'] for ent in observations['entities'] if ent['name']=='CS175CreeperSurviver'][0]
            for ent in observations['entities']:
                if ent['name'] == 'Creeper':
                    obs[0][0][round(ent['z']-agent_Z)+5] = 2  ## 2 代表这个格子里有creeper

            print(obs) ##这里是agent前后creeper的状况 print

            ##这里根据生命值修改score，xml里没有找到直接改的地方
            if observations['Life'] < life:
                observations['Score'] -= (life - observations['Life']) * DAMAGE_TO_SCORE_RATE
            
            ##修改成新的life值
            life = observations['Life'] 

            # Rotate observation with orientation of agent
            yaw = observations['Yaw']
            if yaw == 270:
                obs = np.rot90(obs, k=1, axes=(1, 2))
            elif yaw == 0:
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif yaw == 90:
                obs = np.rot90(obs, k=3, axes=(1, 2))
            
            break

    return obs, life


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
    q_network = QNetwork((2, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network = QNetwork((2, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
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
    life = 20  ##这里life变量记录每次agent的血量，从而算出每次掉血多少

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
        obs, life = get_observation(world_state, life)  ##这里pass进去life变量，同时return了life变量来修改life
        ## 这里obs里，1代表gunpowder，2代表creeper，0代表没有东西

        # Run episode
        while world_state.is_mission_running:
            # Get action
            # allow_break_action = obs[1, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 1
            action_idx = get_action(obs, epsilon)
            
            command = ACTION_DICT[action_idx]
            print(command)

            # Take step
            agent_host.sendCommand(command)
            if(command == 'attack 1'):
                agent_host.sendCommand('attack 0')

            # If your agent isn't registering reward you may need to increase this
            time.sleep(.1)

            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            # if episode_step >= MAX_EPISODE_STEPS or \
            #         (obs[0, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 1 and \
            #         obs[1, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 0 and \
            #         command == 'move 1'):
            #     done = True
            #     time.sleep(2)  
            if episode_step >= MAX_EPISODE_STEPS:   ##重写了这里，comment掉了上面
                done = True
                time.sleep(2)

            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs, life = get_observation(world_state, life)  #这里同样加了life parameter 和 life 的return

            # Get reward
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
            episode_return += reward

            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

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