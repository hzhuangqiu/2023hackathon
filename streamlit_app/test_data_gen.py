import pandas as pd
data = pd.read_csv('../preprocessed_dataset.csv')
data = data.drop(["Index", "Month", "Day", "Time of Day", "Source"], axis=1)
data = data.drop("Target", axis=1)
data = data[:100]
data.to_csv('test_data.csv', index=False)
