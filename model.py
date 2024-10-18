import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


data = pd.read_csv('sample_insurance_data.csv')

data['Premium'] = pd.to_numeric(data['Premium'], errors='coerce')


X = data.drop(['Premium', 'First_Name', 'Last_Name', 'Address'], axis=1)  
y = data['Premium']


X = X.loc[y.index]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numerical_features = ['Age', 'Vehicle_Age', 'Accidents']
categorical_features = ['DL_Class', 'Sex', 'State'] 


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
test_score = model.score(X_test, y_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print(f'Test Score: {test_score:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')


joblib.dump(model, 'insurance_premium_model.pkl')
