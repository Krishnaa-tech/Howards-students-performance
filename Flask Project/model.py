import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pickle import dump

df = pd.read_csv("Student Info.csv")

class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output

cat_features = [feature for feature in df.columns if df[feature].dtype in ['object', 'bool_']]
multi = MultiColumnLabelEncoder(columns=cat_features)
df = multi.fit_transform(df)

dump(multi, open('label_encoder.pkl', 'wb'))

X = df[['age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',
        'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup',
        'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
        'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]
Y = df[['G3']]

X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=49)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("GradientBoostingRegressor")
print("rmse: ", np.sqrt(mse))
print("r2 score: ", r2)

dump(model, open('model.pkl', 'wb'))
