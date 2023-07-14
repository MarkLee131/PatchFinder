import pandas as pd
from sklearn.model_selection import train_test_split
# Load your data from the CSV file
data = pd.read_csv('/mnt/local/Predict5/data1/warnings.csv')

# Split the data into features (X) and labels (y)
X = data.drop('Label', axis=1)  # Replace 'label_column' with the column name of your labels
y = data['Label']               # Replace 'label_column' with the column name of your labels

# Split the data into train and test sets (80% train, 20% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=3407)

# Split the remaining data into validation and test sets (10% validation, 10% test)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=3407)

# Now you have your train, validation, and test sets
# Combine the features and labels for each set
train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the sets as CSV files
train_data.to_csv('/mnt/local/Predict5/data1/train.csv', index=False)
val_data.to_csv('/mnt/local/Predict5/data1/valid.csv', index=False)
test_data.to_csv('/mnt/local/Predict5/data1/test.csv', index=False)
