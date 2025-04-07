# Hidden Goal Recognition (GR) in Overcooked

Contents:
- [Introduction](#introduction)
- [Hungry-Thirsty Environment](#hungry-thirsty-environment)
- [Installation](#installation)
- [Experiments and Customization](#experiments-and-customization)
- [Agents](#agents)

## Introduction

The code here is derived from "Too many cooks: Bayesian inference for coordinating multi-agent collaboration" ([[Full paper]](https://arxiv.org/abs/2003.11778) [[Journal paper]](https://onlinelibrary.wiley.com/doi/10.1111/tops.12525) [[Video]](https://www.youtube.com/watch?v=Fd4RcVaNthY&ab_channel=RoseWang)). Our extensions allow users to implement a "Hungry-Thirsty" environment in Overcooked, where a "fetcher" agent is tasked with identifying the goal of a "human" agent based on the human's movements.

##  Hungry-Thirsty Environment
The Hungry-Thirsty environment is designed for exactly 2 agents: one "fetcher" agent and one "human" agent. The "fetcher" agent is tasked with recognizing what food item (sushi, water, egg, or bread) the "human" agent wants based on the "human" agent's movements. At timestep 0, the "fetcher" agent calculates the shortest walkable path from the initial location of the "human" to all food items in the environment. In subsequent timesteps, the "fetcher" agent computes the total change in shortest walkable path from the "human" to all food items in the environment. If there is more negative change in shortest walkable path for one food item compared to the others, then the "human" is approaching that food item, indicating that it is the item that they want. The "fetcher" then fetches that item and delivers it to the "human". The below [**Youtube Video:**](https://youtu.be/JN4FOKtD4w0) provides examples (note that the woman agent wearing blue is the "fetcher" and the man agent wearing pink is the "human"). 

<p align="center">
  <a href="https://youtu.be/JN4FOKtD4w0">
    <img width="400" alt="Image" src="https://github.com/user-attachments/assets/bdb0834f-0fd7-4d32-85e6-36d31e560d32" />
  </a>
</p>

## Installation

You can install the dependencies with `pip`:
```
git clone https://github.com/kiringodhwani/HiddenGR-Overcooked.git
cd gym-cooking
pip install -e .
```

## Experiments and Customization

Here, we discuss how to customize and run your own Hungry-Thirsty experiments. For the code below, make sure that you are in **gym-cooking/gym_cooking/**. This means, you should be able to see the file `main.py` in your current directory.

### Running an experiment

To run a Hungry-Thirsty experiment, run the following command in **gym-cooking/gym_cooking/** :

`python main.py --num-agents <number> --level <level name> --model1 bd --model2 bd --record`

where `<number>` is the number of agents interacting in the environment (the Hungry-Thirsty environment is designed for 2 agents) and `<level name>` is the name of the level under the directory gym-cooking/gym_cooking/utils/levels, omitting the .txt. Take the below command as an example:

`python main.py --num-agents 2 --level DesignOne --model1 bd --model2 bd --record`

This runs a Hungry-Thirsty experiment with two agents (agent-1 = "fetcher", agent-2 = "human") on the 'DesignOne.txt' level. It also **records** the experiment, such that you can watch a replay afterwards. Please ignore the model selections ("--model1 bd --model2 bd"). Right now, we hard code agent-1 as the "fetcher" and agent-2 as the "human" in the **initialize_agents()** function in **main.py** (gym-cooking/gym_cooking/main.py).

If you would like an agent to use Bayesian Delegation instead of being a "fetcher" or "human", then you must edit the initialize_agents() function in main.py to define said agent as a **RealAgent**. For the different agent definitions, please see gym-cooking/gym_cooking/utils/agent.py as well as the [Agents](#agents) section of this README. 

### Customization

To customize the level for the experiment, please see [**the Design and Customization section of the original Overcooked repo**](https://github.com/rosewang2008/gym-cooking/blob/main/docs/design.md). This page explains how to customize your own level, and it explains how to create your own recipes with your own ingredients. 

### Testing a Level (Manual Control)

To explore a level (environment) that you are building, run the below command in **gym-cooking/gym_cooking/** :

`python main.py --num-agents <number> --level <level name> --play`

This will open up the environment in Pygame. You can manually control the agents using the arrow keys. Only one agent can be controlled at a time, but you can toggle between agents by pressing 1, 2, 3, or 4 (up until the actual number of agents defined in the level). Hit the Enter key to save a timestamped image of the current screen to misc/game/screenshots.

### Watching the Experiment Replay

When you specify the --record flag in your command to run the experiment, you can watch a replay/recording of the experiment after it finishes. To do so, navigate to **gym-cooking/gym_cooking/play_video.py**, edit the directory containing the image files for the experiment, and then run `python play_video.py` in terminal to start the replay. Click any key to flip through the replay from start to finish. For clarity on this step, please watch the tutorial video below. 

### Tutorial Video

Below is a [**demo video**](https://youtu.be/f6NuPhPtyD0) showing the entire process of creating and running your own experiment. The video shows how to customize your own level/environment, manually test the environment, run an experiment in the environment, and watch a replay of the experiment...

[<img width="1416" alt="Image" src="https://github.com/user-attachments/assets/5fb9775e-8649-4459-84fe-d39a85dc8d7f" />](https://youtu.be/f6NuPhPtyD0)

## Agents

Here, I will discuss the differences between the various types of agents defined in the **agent.py** file (gym-cooking/gym_cooking/utils/agent.py). I will also provide a template for creating your own types of agents and explain how to run experiments with them. 

### Physical Agents vs. Decision-Making Agents

The original Overcooked code implements a clear separation between agents that make decisions (RealAgent, HybridAgent, FetchingAgent) and agents that represent the physical entities in the simulated world (SimAgent).

**SimAgent**: SimAgents are the physical representation of agents in the environment.
- Track physical properties: location, what the agent is holding, current action
- Handle physical interactions with the world (movement, object acquisition/release)
- Used by the environment to manage collisions and execute actions

**RealAgent**, **HybridAgent**, **FetchingAgent** (decision-making agents): The brains that make decisions.

The use of SimAgents and decision-making agents allows for clean separation of physical simulation and decision-making logic. See the below explanation of how these agents work together: 
1. The environment creates SimAgents for visualization and physics handling. 
2. Outside the environment, decision-making agents (RealAgent, HybridAgent, FetchingAgent) are initialized with the same names.
3. At each timestep:
    * Each decision-making agent receives the current observation (which includes the SimAgents)
    * The decision-making agent finds its corresponding SimAgent in the observation
    * The decision-making agent updates its state based on the SimAgent (location, holding)
    * The decision-making agent decides what action to take
    * The action is passed to the environment, which updates the SimAgent accordingly

### RealAgent vs. FetchingAgent vs. HybridAgent 

**RealAgent** (from [the original Overcooked code](https://github.com/rosewang2008/gym-cooking/blob/main/gym_cooking/utils/agent.py)):
- Use Bayesian Delegation (as defined in this paper) to rapidly infer the hidden intentions of others by inverse planning. 
- Navigation planning using BRTDP (Bounded Real-Time Dynamic Programming)

**FetchingAgent**: This is the "**fetcher**" in our experiments. Represents a specialized assistant that observes another agent's behavior (the "human" agent) to determine what object to fetch (sushi, water, egg, or bread), fetches said object, and delivers it to the other agent ("human" agent). Uses a state machine with states: OBSERVE, FETCH, DELIVER, RETURN, and DONE…
- OBSERVE: Monitors the human agent's movement patterns and calculates the total change in shortest walkable path from the human agent to every target object (Sushi, Water, Egg, Bread) to determine which object the human is trying to reach.
- FETCH: Navigates through the environment using pathfinding to locate and pick up the identified target object.
- DELIVER: Carries the retrieved object to the human agent and transfers it when adjacent.
- RETURN: Returns to its original starting position after successfully delivering the object.
- DONE: Remains stationary at its original position, indicating successful completion of the fetch-and-deliver cycle.

**HybridAgent**: This is the "**human**" in our experiments. Simulates a human that starts with simple behavior navigating towards its target item (sushi, water, egg, or bread) and then switches to more complex planning (RealAgent) that responds to the behavior of the "fetcher" agent. Uses a state machine with states: SIMPLE and REAL_AGENT…
- Begins in "SIMPLE" mode with direct path planning to its target item (sushi, water, egg, or bread).
- Transitions to "REAL_AGENT" mode when the fetcher agent moves.
- When transitioning, it creates a full RealAgent internally and delegates decisions to it.

### Decision-Making Agent Template 

All three decision-making agents I presented (RealAgent, HybridAgent, FetchingAgent) follow a similar template despite their different internal logic. Here is a template for creating a similar decision-making agent that functions in the environment:

1. **Initialize with required parameters**:
- Purpose: Sets up the agent's identity and configuration
- Key parameters:
    - name: Unique identifier for the agent (e.g., "agent-1")
    - color: Display color for visualization
    - arglist (not required, fetcher doesn't use): Command-line arguments with environment settings
    - recipes (not required, fetcher doesn't use): Available food recipes the agent can prepare

```
def __init__(self, name, color, arglist=None, recipes=None):
    self.name = name
    self.color = color
    # Store other required attributes
    
```

2. **Implement the select_action method**:
- Purpose: The core decision-making function called at each timestep to determine how the agent will act.
- Input: Current environment observation containing world state and other agents.
- Process:
    1. Identifies the agent's physical representation in the simulation (SimAgent)
    2. Updates internal state based on simulation
    3. Runs decision-making logic to determine next action
- Output: Returns a directional action as (dx, dy) tuple such as (0,1), (-1,0), etc.

```
def select_action(self, obs):
    # Find your SimAgent representation in the observation
    sim_agent = next((a for a in obs.sim_agents if a.name == self.name), None)
    
    # Update internal state from SimAgent
    self.location = sim_agent.location
    self.holding = sim_agent.holding
    self.action = sim_agent.action
    
    # Determine next action based on internal logic
    # ...
    
    return action  # Return a (dx, dy) tuple like (0, 0), (1, 0), etc.
```

3. **Implement the required metrics tracking interface**: These are required methods for compatibility with the environment metrics defined in the original Overcooked code (i.e., allow the environment to track agent progress).

```
def refresh_subtasks(self, world):
    # Update subtask completion state (i.e., updates completion status of current goals)
    pass
    
def get_action_location(self):
    # Return where agent will be after taking action
    return tuple(np.asarray(self.location) + np.asarray(self.action))
    
def all_done(self):
    # Reports if agent has completed all assigned tasks
    return False
```

4. **Implement necessary utility methods*: 

**get_holding** – standardize how held objects are reported

```
def get_holding(self):
    # Return name of held object for consistency
    if self.holding is None:
        return 'None'
    return self.holding.full_name
```

**__str__** – create visual representation of agent for the display

```
def __str__(self):
    # For visualization
    from termcolor import colored as color
    return color(self.name[-1], self.color)
```

### Defining Agent Types in Your Experiments

To choose the decision-making agents used in your experiments, you will need to use the ***initialize_agents()*** function in **main.py** (gym-cooking/gym_cooking/main.py). In the below example, my level for the experiment consists of two agents, and I make agent-1 a FetchingAgent ("fetcher") and agent-2 a HybridAgent ("human"):

```
def initialize_agents(arglist):

    . . .

    if len(real_agents)+1 == 1:
        # MAKE AGENT 1 A "FETCHER"
        print(f'Initializing FetchingAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
        real_agent = FetchingAgent(
                        arglist=arglist,
                        name='agent-'+str(len(real_agents)+1),
                        color=COLORS[len(real_agents)])
    else:
        # MAKE AGENT 2 A "HUMAN"
        print(f'Initializing HybridAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
        real_agent = HybridAgent(
                        arglist=arglist,
                        name='agent-'+str(len(real_agents)+1),
                        id_color=COLORS[len(real_agents)],
                        recipes=recipes)
```

















