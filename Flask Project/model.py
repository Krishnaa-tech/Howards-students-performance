# import pandas as pd
# import numpy as np
# from pickle import load
# from sklearn.preprocessing import LabelEncoder

# # Load encoder, scaler, and model
# encoder = load(open('label_encoder.pkl', 'rb'))
# model = load(open('model.pkl', 'rb'))

# # Extracting the relevant features from the DataFrame
# features = ['Age', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

# # Creating an empty dictionary to store user input
# user_input = {}

# # Take input from the user for each feature
# for feature in features:
#     user_input[feature] = float(input("Enter {}: ".format(feature)))  # Convert input to float

# # Creating a DataFrame with the user input
# user_df = pd.DataFrame([user_input])

# # Calculating 'avg_grade' based on 'G1', 'G2', and 'G3'
# user_df['avg_grade'] = user_df[['G1', 'G2', 'G3']].mean(axis=1)

# # Display the user input DataFrame
# print("\nUser Input:")
# print(user_df)

# # Label encoding for categorical columns
# categorical_columns = ['Mjob', 'Fjob']

# label_encoder = LabelEncoder()
# for column in categorical_columns:
#     user_df[column] = label_encoder.fit_transform(user_df[column])



# # Display the preprocessed user input DataFrame
# print("\nPreprocessed User Input:")
# print(user_df)




import pandas as pd
from pickle import load
from sklearn.preprocessing import LabelEncoder

# Load encoder, scaler, and model
encoder = load(open('label_encoder.pkl', 'rb'))
model = load(open('model.pkl', 'rb'))

# Extracting the relevant features from the DataFrame
features = ['Age', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

# Creating an empty dictionary to store user input
user_input = {}

# Take input from the user for each feature
for feature in features:
    if feature in ['Mjob', 'Fjob']:
        user_input[feature] = input("Enter {}: ".format(feature))
    else:
        user_input[feature] = float(input("Enter {}: ".format(feature)))  # Convert input to float for numerical features

# Creating a DataFrame with the user input
user_df = pd.DataFrame([user_input])

# Calculating 'avg_grade' based on 'G1', 'G2', and 'G3'
user_df['avg_grade'] = user_df[['G1', 'G2', 'G3']].mean(axis=1)

# Display the user input DataFrame
print("\nUser Input:")
print(user_df)

# Label encoding for categorical columns
categorical_columns = ['Mjob', 'Fjob']

label_encoder = LabelEncoder()
for column in categorical_columns:
    user_df[column] = label_encoder.fit_transform(user_df[column])

# Display the preprocessed user input DataFrame
print("\nPreprocessed User Input:")
print(user_df)
