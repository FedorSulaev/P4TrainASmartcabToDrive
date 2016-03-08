import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
import matplotlib.pyplot as plt

State = namedtuple("State", 
        ["light",
        "oncoming",
        "right",
        "left",
        "next_waypoint"])
possible_actions = (None, "forward", "left", "right")

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.encountered_states = {}
        self.epsilon = 0.1
        self.alpha = 1.0
        self.rewards = []
        self.average_rewards = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = State(
            light=inputs["light"], 
            oncoming=inputs["oncoming"],
            left=inputs["left"], 
            right=inputs["right"],
            next_waypoint=self.next_waypoint)
        
        # TODO: Select action according to your policy
        first_time = not self.encountered_states.has_key(self.state)
        if first_time or random.random() < self.epsilon:
            action = random.choice(possible_actions)
        else:
            action_estimates = self.encountered_states[self.state]
            action = max(action_estimates, key=lambda x: action_estimates[x])

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.rewards.append(reward)
        self.average_rewards.append(sum(self.rewards) / float(len(self.rewards)))

        # TODO: Learn policy based on state, action, reward
        if first_time:
            self.encountered_states[self.state] = {}
            for a in possible_actions:
                self.encountered_states[self.state][a] = 0.0
        current_estimate = self.encountered_states[self.state][action]
        new_estimate = current_estimate + self.alpha * (reward - current_estimate)
        self.encountered_states[self.state][action] = new_estimate

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
 
    plt.plot(a.average_rewards)
    plt.title("alpha = {}, epsilon = {}".format(a.alpha, a.epsilon))
    plt.xlabel("Turn")
    plt.ylabel("Average reward")
    plt.show()


if __name__ == '__main__':
    run()
