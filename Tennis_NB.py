#Malak Nassar
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Create a DataFrame with the provided data
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Encode categorical variables using LabelEncoder
label_encoders = {}
for column in data.columns:
    if data[column].dtype == "object":
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Specify the target variable column name
target_column_name = "Play Tennis"

# Split the dataset into features (X) and the target variable (y)
X = data.drop(target_column_name, axis=1)
y = data[target_column_name]

# Create a Gaussian Naive Bayes classifier
clf = GaussianNB()

# Train the Naive Bayes classifier
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)

# Print the predictions
print("Predictions:")
for i, prediction in enumerate(predictions):
    print(f"Sample {i + 1}: Predicted = {prediction}, Actual = {y.iloc[i]}")