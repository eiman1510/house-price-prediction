import pandas as pd
import yaml
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# 1. Load Data
df = pd.read_csv('data/House_Prices_Data.csv')

# 2. Simple Preprocessing
# For simplicity, we'll use a few numerical columns and fill missing values
df['Area Size'] = pd.to_numeric(df['Area Size'], errors='coerce')
features = ['bedrooms', 'baths', 'Area Size']
target = 'price'
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# 3. Split Data
split_params = params['train_test_split']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=split_params['test_size'],
    random_state=split_params['random_state']
)

# 4. Train Model
model_params = params['model_params']
model = RandomForestRegressor(
    n_estimators=model_params['n_estimators'],
    max_depth=model_params['max_depth'],
    random_state=model_params['random_state']
)
model.fit(X_train, y_train)

# 5. Evaluate and Save Metrics
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")

with open('metrics.json', 'w') as f:
    json.dump({'r2_score': r2}, f)

# 6. Save Model
joblib.dump(model, 'models/model.joblib')
print("Model trained and saved to models/model.joblib")