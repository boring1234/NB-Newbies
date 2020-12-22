---
layout: default
title:  Final Report
---

## Video

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
