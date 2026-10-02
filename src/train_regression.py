import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import get_preprocessor


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("data/traffic_data_1000_rows.csv")


# ==============================
# 2. Prepare Date
# ==============================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day_of_Week"] = df["Date"].dt.dayofweek

df = df.drop("Date", axis=1)


# ==============================
# 3. Select X and Y
# ==============================

# Vehicle_Count is our target
# Traffic_Level is removed because it
# was created from Vehicle_Count.

X = df.drop(["Vehicle_Count", "Traffic_Level"], axis=1)

y = df["Vehicle_Count"]


# ==============================
# 4. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==============================
# 5. Create Preprocessor
# ==============================

preprocessor = get_preprocessor()


# ==============================
# 6. Create Random Forest
# ==============================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ==============================
# 7. Create Pipeline
# ==============================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==============================
# 8. Train Model
# ==============================

pipeline.fit(X_train, y_train)


# ==============================
# 9. Make Predictions
# ==============================

y_pred = pipeline.predict(X_test)


# ==============================
# 10. Evaluate Model
# ==============================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n==============================")
print("RANDOM FOREST REGRESSION")
print("==============================")

print("\nMean Absolute Error:")
print(mae)

print("\nRoot Mean Squared Error:")
print(rmse)

print("\nR2 Score:")
print(r2)


# ==============================
# 11. Save Model
# ==============================

joblib.dump(
    pipeline,
    "models/traffic_regressor.pkl"
)

print("\nRegression model saved successfully!")
print("Location: models/traffic_regressor.pkl")