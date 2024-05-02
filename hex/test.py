import matplotlib.pyplot as plt

# Provided data
data = {
    'episode_0': 204,
    'episode_10': 254,
    'episode_20': 155,
    'episode_30': 100,
    'episode_40': 400,
    'episode_50': 387
}

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(data.keys(), data.values(), color='skyblue')
plt.xlabel('Episodes')
plt.ylabel('Wins')
plt.title('Wins per Episode')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
