import pandas as pd
import matplotlib.pyplot as plt

# File paths
files = [
    "logs/CNN_PPO_0_distractors_2024-05-27_00-58-00.csv",
    "logs/ORACLE_0_distractors_2024-05-27_19-27-58.csv",
    "logs/PRE_TRAINED_0_distractors_2024-05-27_00-57-14.csv"
]

# Reading the CSV files
dfs = [pd.read_csv(file) for file in files]

# Displaying the first few rows of each DataFrame to understand their structure
dfs_info = [df.head() for df in dfs]
dfs_info


# Separate the dataframes by their types
cnn_dfs = dfs[0]
oracle_dfs = dfs[1]
pretrained_df = dfs[2]

# Concatenate them separately
cnn_df = cnn_dfs
oracle_df = oracle_dfs
pretrained_df = pretrained_df  # Already a single dataframe

# Plot Mean Reward over Num Timesteps for each type
plt.figure(figsize=(10, 6))
plt.plot(cnn_df['Num Timesteps'], cnn_df['Mean Reward'], label='CNN', color='blue', linewidth=0.5)
plt.plot(oracle_df['Num Timesteps'], oracle_df['Mean Reward'], label='Oracle', color='red', linewidth=0.5)
plt.plot(pretrained_df['Num Timesteps'], pretrained_df['Mean Reward'], label='Pretrained', color='green', linewidth=0.5)
plt.xlabel('Num Timesteps')
plt.ylabel('Mean Reward')
plt.title('Mean Reward over Num Timesteps for CNN, Oracle, and Pretrained')
plt.legend()
plt.grid(True)
plt.show()