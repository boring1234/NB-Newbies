---
layout: default
title:  Final Report
---

## Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/gDRtSZjo7Ok?ecver=1" frameborder="0" allowfullscreen></iframe>
## Project Summary
#### Half Quarter Report
Our project focuses on making the agent survive as long as possible during a creeper attack. The agent will be born in a random place on the platform and after 5 seconds the creepers will start emerging from random places on the map at a certain rate. The agent will be given information about the location of the creepers that exist around the agent. The agent needs to learn how to make the creepers explode and not hurt itself. The project can be divided into two parts. At first, we train the agent in a 1-dimensional graph. The agent can only move forward and backward and attack. Then, we implement it in a 2-dimensional graph.

#### Final Quarter Report

<!---Use another level-two header to start a Project Summary section. Write a few paragraphs summarizing the goals of the project (yes, yet again, but updated/improved version from the status). In particular, make sure that the problem is clearly deﬁned here, and feel free to use an image or so to set up the task. Part of the evaluation will be on how well you are able to motivate the challenges of the problem, i.e. why is it not trivial, and why you need AI/ML algorithms to solve it.--->

The goal of our project is to create an agent that is capable of surviving from the explosion of creepers in Minecraft. In Minecraft, a creeper is a harmful mob that approaches players silently. The only way that a creeper can harm players is to explode suicidally near the players. Players can survive and even stay full blood by being far away from the explosion. For human players, it is very easy to escape a creeper’s explosion. When a creeper is about to explode, it will be standing still and changing its color back and forth for seconds, and we humans can pick up these signs and just run away before the explosion. However, it is a difficult task for a computer to do all these steps. 

In the early stage of the project, we used a vanilla Deep Q-network reinforcement learning algorithm to train our agent. The result of the training was very disappointing because we did not see expected behaviors from our agent. Our agent most of the time was not able to escape from the explosion completely, always resulting hurted and nearly dead. Sometimes the agent would do the opposite by running toward creepers when the creepers were about to explode. We eventually switched to the PPO (Proximal Policy Optimization) reinforcement learning algorithm, which significantly improved our agent’s performance. We were able to visually confirm that our agent became smarter by triggering creepers to explode while running away with full blood.


## Approaches
#### Half Quarter Report
The machine learning algorithms we used in this project is heavily based on Assginment 2, CS 175 fall 2020. We used multi-layer neural network and e-greedy learning policy to train our agent. The agent takes in a 1×1×11 matrix, and it returns an integer value that represents the next move it takes among {moving forward, moving backward, attacking}.
<br>
By using the e-greedy policy, our agent chooses either to explore or to exploit according to a epsilon value growing by each iteration as well as a constant value R where 0<R<1. The activation functions (or layers) in our neural network model is still naive at this point. More trials would be needed before we decide what functions best fit our neural network.
<br>
To evaluate the reward, we record the health power (HP) of the character at the end of each iteration. Moves that maximize the remaining HP would be prefered by our algorithm.

#### Final Quarter Report
First, we decided to start our project on a 1-dimensional map. The agent can only move forward, move backward and attack and one creeper will appear 5 blocks away from the agent. We want to see that if the agent can successfully learn how to get away from the creepers after training by q-network and sequential model. In this approach, our approach set is as following(Observation, Reward, Action, Terminal). 

	Episode: Play through the environment
		Survive from Creepers(1-d)
	Observation:
		get_observation(world_state)
		zero numpy array(1, 1, 11)
		channel 0: NOTHING
		channel 1: Gunpowder(not use anymore at last)
		channel 2: Creeper
	Action
		Discrete(4)
		move forward, move backward, attack, stop attacking
	Reward
		+1 if Creepers is more than 3 blocks away from agent
		-1 if Creepers is within 3 blocks away from the agent
	Terminal
		After 20 steps or die from explosion of Creepers

In this approach, we decided to make our observation 5 blocks backward and frontward from the agent because the Creepers will be attacked only if the agent is within 5 blocks away from them and we want to minimize our observation size to decrease the running time and avoid unnecessary running. We decided to record three types of things on the floor(Creepers, Nothing, Gunpowder). We want the Creepers and Nothing to calculate the distance from the Creepers and we want the gunpowder to record how many Creepers the agent can learn to kill after training. However, we found that the Gunpowder does not really work for the 1-d dimension because first, it is too hard for an agent to learn how to kill creepers (even hard for a human player) within such a short term of training(We set our total step to 4000 for this approach), and second it is also inaccurate to count the killed Creepers by this because the agent might not observe the gunpowder if he moves away after killing the Creepers and other Creepers explosion can destroy the gunpowder as well. But we want to keep this for a 2-d version because if we increase the training time and steps for every episode, we might get better results from this. Also, for the attack action, we met some difficulties because the agent keeps destroying the floor. After we check the Malmo mission handler file, we find that we need to add attack 0(stop attacking) into the action space. However, here we force to send attack 0 commands after sending the attack 1 command because it will only be usable if the agent is attacking. 
The result of this approach is as follow:

<img src="https://raw.githubusercontent.com/boring1234/NB-Newbies/main/docs/1.JPG" width="560"/>

As we can see, The reward increase from an average of 3 to 7 after around 3000 steps training, which is a pretty small training and a good result we think. 

After implementing our algorithm through Q-network, we decided to change our algorithm to RLlib, using gym library instead. The reason that we use RLlib is because of the suggestion of our TA and he told us that deep reinforcement learning would train the agent better than Q-network. For the 1-d rllib version project, we did not train it because we want to save time and move on to the 2-d version. 

For our third approach, we decided to use 2-d map for our project using q-learning(The reason why we did not implemented it using RLlib is that we wrote 1-d RLlib and 2-d q-network at the same time). This time our approach set changed to this:

	Episode: Playthrough of the environment
		Survive from Creepers
	Observation:
	get_observation(world_state)
		zero numpy array(1, 11, 11)
		channel 1: NOTHING
		channel 2: Gunpowder(not use anymore at last)
	channel 3: Creeper
	Action
		Discrete(6,)
		forward, backward, turn left, turn right, attack, not attack
		channel one: 0: move 1, 1: move 0
		channel two: 0: turn 1, 1: turn -1
		channel three: 0: attack 1, 1: attack 0
	Reward
		+7 if agent's health level is above 15
		-1.5 if agent's health level is below 15, above 10
		-3 if agent's health level is below 10
		-20 if agent died
	Terminal
		After 100 steps or die from explosion of Creepers

In this approach set, first, we keep the gunpowder here to see if it is useful or not. For the action space, we add turn 1 and turn -1(turn left and turn right) to the action set so that the agent can move to anywhere he wants. This time, we changed our reward based on the health level of the agent. If the agent has a high health level, he will get a positive reward and if he has a low health level, we will deduct the reward. If the agent died, it will get a really bad reward. We choose this health because based on the wiki of Minecraft, the average damage of creeper is around 5. This time, we get a pretty good result after around 120000 steps as shown below:

<img src="https://raw.githubusercontent.com/boring1234/NB-Newbies/main/docs/2.JPG" width="560"/>

The reward change from -25 to 0 after around 120000 steps of training. When we try to print out the gunpowder data in each episode, we found that the number is always 0 and as we checked the agent movement at the end of the training, we found that it is impossible for the agent to learn how to kill a creeper since it is too hard for him and we also cannot make sure the agent can observe the gunpowder if there is one. Thus, we decided to delete gunpowder. Besides, because we think that the agent cannot kill the creepers, we also decided to delete the attack action here. Finally, for the reward, because we are trying to improve our learning, we communicated with our TA and we think that it will be better if the reward can dynamically change based on the damage it gets from every step. 

In our next approach, instead of changing the approach set, we also decided to use rllib to improve our training. Besides, we also found a problem while running the previous discrete movement settings. The damage reward attribute in XML does not really work because the reward and observationFromRay are not working for the discrete setting. So we decided to move on to a continuous setting for the agent. This will also let the agent move freely in the Minecraft world without bounding by the block in it. Because the Creepers is a continuous setting, it will also make the agent move faster to get away from Creepers and improve learning in some way. Our approach set is as follows:

	Episode: Playthrough of the environment
		Survive from Creepers
	Observation:
		get_observation(world_state)
		zero numpy array(1, 11, 11)
		channel 1: NOTHING
		channel 2: Creeper
	Action
		Continuous(4,)
		forward, backward, turn left, turn right
		channel one: 0: move 1, 1: move 0
		channel two: 0: turn 1, 1: turn -1
	Reward
		+1 for no damage and start healing
		-X (X=damage the agent get from creepers) for damage from creepers, scaled by health damage the agent get

		-10(extra) if the agent died, it will get a 10 extra deduction of the reward
	Terminal
		After 100 steps or die from the explosion of Creepers

As you can see, we changed our approach set as above. This time, our project is getting closer to our final stage and here is how our final code work here. 
self.action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
self.observation_space = Box(0, 1, shape=(np.prod([1, 2*self.obs_size+1, 2*self.obs_size+1]), ), dtype=np.int32)
The action_space will have actually two channels and each will three parameters: -1, 0, 1 represent move -1, move 0 and move 1. The observation_space will have two observation: 0 for nothing and 1 for creepers. And for our step and observation function, here is the pseudo code:

	Def step(action):
		send command: action[0] move
		send command: action[0] turn
		get the observation
		check if the episode is finished: done
		get the reward from XML
		if current_life >= previous_life:
			reward += 1
		else:
			reward -= (previous_life - current_life)
		if current_life == 0:
			reward -= 10

Def get_observation(world_state):
initialize observation size: [1, 11, 11]
check if the world state is running
find the location of the creepers and the agent based on the information in the world state
calculate which block the creepers and agent is in and assign the creepers into the observation
get the 'Yaw' data and change the observation based on the number of Yaw

However, this time, we did not get what we expected:

<img src="https://raw.githubusercontent.com/boring1234/NB-Newbies/main/docs/3.JPG" width="560"/>

We expected the learning to be much better, however, it is not better than the previous learning. We summarize what we changed in this approach and we found that it can only because we delete the attack action. And then, we checked online and we found that if the creepers are attacked by the agent, the creeper will start to explode without being within 1 block away from the agent. Thus, at last, we decided to add back attack action. Besides, we also add some parameters in the return text and png file. We added the number of the steps for each episode, the number of creepers that have chased the agent for each episode, and the steps that the agent moved with creepers around him. These data can help us find that if our train is working or not. Our final stage approach set is as follows: 

	Episode: Playthrough of the environment
		Survive from Creepers
	Observation:
		get_observation(world_state)
		zero numpy array(1, 11, 11)
		channel 1: NOTHING
		channel 2: Creeper
	Action
		Continuous(4,)
		forward, backward, turn left, turn right, attack, stop attacking
	Reward
		+1 for no damage and start healing
		-X (X=damage it get from creepers) for damage from creepers, scaled by health damage it get

		-10(extra) if the agent died, it will get a 10 extra deduction of the reward
	Terminal
		After 100 steps or die from the explosion of Creepers

We also changed the send command code like this:

	if self.allow_attack_action and action[2]>0:
		self.agent_host.sendCommand('move 0')
		self.agent_host.sendCommand('turn 0')
		self.agent_host.sendCommand('attack 1')
		time.sleep(1)
	else:
		self.agent_host.sendCommand('attack 0')
		self.agent_host.sendCommand('move {:30.1f}'.format(action[0]))
		self.agent_host.sendCommand('turn {:30.1f}'.format(action[1]))
		time.sleep(.2)

This time, we need to consider if we need to send the attack 0 commands or not. Our result is as follows:

<img src="https://raw.githubusercontent.com/boring1234/NB-Newbies/main/docs/4.PNG" width="560"/>

AND 

<img src="https://raw.githubusercontent.com/boring1234/NB-Newbies/main/docs/5.PNG" width="720"/>

As you can see we have a pretty good result in around 250000 steps. We also can see that the steps that the agent can survive for every episode increase a lot. The data for the steps that the agent followed by creepers also increase and this data will help us ignore the situation that the agent moved around with no creepers around. 

Well, Finally, as shown in the video below, we found that the agent also learns something amazing: the agent will Wait for the creepers to get closer and move away immediately, which is the best way to make the creeper explode without hurt himself. 

<iframe width="560" height="315" src="https://youtu.be/7WoGPztsoSg" frameborder="0" allowfullscreen></iframe>

<!---Use another level-two header called Approaches, In this section, describe both the baselines and your proposed approach(es). Describe precisely what the advantages and disadvantages of each are, for example, why one might be more accurate, need less data, take more time, overﬁt, and so on. Include enough technical information to be able to (mostly) reproduce your project, in particular, use pseudocode and equations as much as possible.--->

## Evaluation
#### Half Quarter Report
###### Qualitative
Up till the status report, we ensure that the program runs as expected, while the algorithm takes in accurate inputs (e.g. observation) and executes correctly. However, since we have not set up the reward calculation, we cannot output total reward after each iteration. At this point, any sequence of actions would result in 0, and thus make every move the same given each observation. This issue will be fixed after we fully construct our Q-network and handle the health power (HP) calculation.

###### Quantitive
<img src="https://raw.githubusercontent.com/boring1234/NB-Newbies/main/docs/result.jpg"/>
<br>
The current result of our algorithm does not show any sign of learning. As we clarified in qualitive evaluation, one foundational problem is that we have not yet quantified the character's HP as the reward for ML algorithm. It results in 0 because nothing is counted towards the total reward at the end of each iteration. After the status report, we will update the reward calculation so that the agent can start learning.

#### Final Quarter Report

<!---An important aspect of your project, as I’ve mentioned several times now, is evaluating your project. Be clear and precise about describing the evaluation setup, for both quantitative and qualitative results. Present the results to convince the reader that you have solved the problem, to whatever extent you claim you have. Use plots, charts, tables, screenshots, ﬁgures, etc. as needed. I expect you will need at least a few paragraphs to describe each type of evaluation that you perform.--->

## References
Source code of this project refer to Assginment 2, CS 175: Project in AI, Fall 2020, UCI
<br><br>
Minecraft platform supported by Malmo, Microsoft
<br>
https://github.com/microsoft/malmo
<br><br>
Epsilon Greedy Strategy refers to "Exploration vs. Exploitation - Learning the Optimal Reinforcement Learning Policy", deeplizard
<br>
https://www.youtube.com/watch?v=mo96Nqlo1L8&t=30s
<br><br>
Neuro Network architecture refers to "Building our Neural Network - Deep Learning and Neural Networks with Python and Pytorch p.3", sentdex
<br>
https://www.youtube.com/watch?v=ixathu7U-LQ&feature=youtu.be&t=210
