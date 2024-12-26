# This python script creates a plot of the average reward against the number of episodes


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/ameya/turtlebot_ws2/src/td3_planning/Rewards_3.csv', header=None)

data = data.iloc[0]  # Extract the first (and only) row

data = pd.DataFrame({
    'episode': data.index + 1,
    'average_reward': data.values
})

plt.figure(figsize=(10, 6))
plt.plot(data['episode'], data['average_reward'], color='b', label='Average Reward')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward vs. Number of Episodes')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('average_reward_vs_episodes.png', format='png', dpi=300)
