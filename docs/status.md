---
layout: default
title:  Status
---
# Video Summary

Video link: https://youtu.be/uHg_NpBoVXQ

<iframe width="560" height="315" src="https://www.youtube.com/embed/uHg_NpBoVXQ?ecver=1" frameborder="0" allowfullscreen></iframe>

## Project Summary

Our project focuses on making the agent survive as long as possible during a creeper attack. The agent will be born in a random place on the platform and after 5 seconds the creepers will start emerging from random places on the map at a certain rate. The agent will be given information about the location of the creepers that exist around the agent. The agent needs to learn how to make the creepers explode and not hurt itself. The project can be divided into two parts. At first, we train the agent in a 1-dimensional graph. The agent can only move forward and backward and attack. Then, we implement it in a 2-dimensional graph.

## Approach

The machine learning algorithms we used in this project is heavily based on Assginment 2, CS 175 fall 2020. We used multi-layer neural network and e-greedy learning policy to train our agent. The agent takes in a 1×1×11 matrix, and it returns an integer value that represents the next move it takes among {moving forward, moving backward, attacking}.
<br>
By using the e-greedy policy, our agent chooses either to explore or to exploit according to a epsilon value growing by each iteration as well as a constant value R where 0<R<1. The activation functions (or layers) in our neural network model is still naive at this point. More trials would be needed before we decide what functions best fit our neural network.
<br>
To evaluate the reward, we record the health power (HP) of the character at the end of each iteration. Moves that maximize the remaining HP would be prefered by our algorithm.

## Evaluation

#### Qualitative
Up till the status report, we ensure that the program runs as expected, while the algorithm takes in accurate inputs (e.g. observation) and executes correctly. However, since we have not set up the reward calculation, we cannot output total reward after each iteration. At this point, any sequence of actions would result in 0, and thus make every move the same given each observation. This issue will be fixed after we fully construct our Q-network and handle the health power (HP) calculation.

#### Quantitive
<img src="https://raw.githubusercontent.com/boring1234/NB-Newbies/main/docs/result.jpg"/>
<br>
The current result of our algorithm does not show any sign of learning. As we clarified in qualitive evaluation, one foundational problem is that we have not yet quantified the character's HP as the reward for ML algorithm. It results in 0 because nothing is counted towards the total reward at the end of each iteration. After the status report, we will update the reward calculation so that the agent can start learning.

## Remaining Goals and Challenges



## Resources Used

