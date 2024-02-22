# Understanding the simple code
# Difference between greedy and epsilon algorithm for multi-arm bandit problem
# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1
EPS_MIN = EPS/1000
EPS_INIT = EPS
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

# Class BanditArm
class BanditArm:
    # Argument p with 3 instance variables
    def __init__(self, p):
        # p: the win rate
        self.p = p # True win rate
        self.p_estimate = 0 # Current estimated win rate
        self.N = 0 # Number of samples collected so far

    # Pull method which returns a 1 with probability p
    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    # Update function which updates our current estimate of this bandit's mean
    def update(self, x):
        self.N += 1 # Increment it as we collected one more sample
        self.p_estimate = (1/self.N) * ((self.N-1) * self.p_estimate + x) # Update the sample mean

# Function to find index with max value (randomized if two or more indices have same max value)
def choose_random_argmax(a):
  idx = np.argwhere(np.amax(a) == a).flatten()
  return np.random.choice(idx)

def experiment(option):
    global EPS
    bandits = [BanditArm(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS) # to store all the rewards for the trials
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0 # number of times we selected the optimal bandit

    # NOTE: Optimal j value is not known in real life
    optimal_j = np.argmax([b.p for b in bandits]) # select index corresponding to the band with the max true mean
    print("optimal j:", optimal_j)
    for i in range(NUM_TRIALS):

        if option == 'Greedy':
            # use greedy to select the next bandit
            num_times_exploited += 1
            #  if there are multiple maximum values in the array, randomly select one of their indices
            j =  choose_random_argmax([b.p_estimate for b in bandits])

        elif option == 'EpsilonGreedy':
            # use  epsilon-greedy to select the next bandit
            if np.random.random() < EPS:
                num_times_explored += 1
                j = np.random.randint(len(bandits)) # Randomly select amongst the bandits
            else:
                num_times_exploited += 1
                #  if there are multiple maximum values in the array, randomly select one of their indices
                j =  choose_random_argmax([b.p_estimate for b in bandits])

        elif option == 'DecayingEG':
            # use decaying epsilon-greedy to select the next bandit

            EPS = max(EPS_INIT - 0.00001 * NUM_TRIALS, EPS_MIN)
            if np.random.random() < EPS:
                num_times_explored += 1
                j = np.random.randint(len(bandits)) # Randomly select amongst the bandits
            else:
                num_times_exploited += 1
                #  if there are multiple maximum values in the array, randomly select one of their indices
                j =  choose_random_argmax([b.p_estimate for b in bandits])


        if j == optimal_j:
            num_optimal += 1

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # print mean estimates for each bandit
    for b in bandits:
        print("mean estimate:", b.p_estimate)

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num_times_explored:", num_times_explored)
    print("num_times_exploited:", num_times_exploited)
    print("num times selected optimal bandit:", num_optimal)

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()


if __name__ == "__main__":
    options = ['Greedy','EpsilonGreedy', 'DecayingEG']
    experiment(options[1])