import pandas as pd

# Load both datasets
app = pd.read_csv("dataset/application_record.csv")
credit = pd.read_csv("dataset/credit_record.csv")

# Flag bad credit behavior
bad_status = ['1', '2', '3', '4', '5']
credit['label'] = credit['STATUS'].apply(lambda x: 0 if x in bad_status else 1)

# For each user, assign label based on worst credit history
credit_labels = credit.groupby('ID')['label'].min().reset_index()

# Merge label with applicant info
data = pd.merge(app, credit_labels, on='ID', how='inner')

# Drop unnecessary columns (optional)
data.drop(['ID'], axis=1, inplace=True)

# Convert all object columns (categorical) to one-hot encoded columns
data = pd.get_dummies(data, drop_first=True)

# Save cleaned dataset
data.to_csv("dataset/credit_data.csv", index=False)
