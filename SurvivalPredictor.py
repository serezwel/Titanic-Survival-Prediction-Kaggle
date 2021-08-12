import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

train_data = pd.read_csv("train.csv")
sns.heatmap(train_data.isna(), cmap='viridis', yticklabels=False, cbar=False)