import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_and_prepare_data():
    # Load dataset
    df = pd.read_csv("data/traffic_data_1000_rows.csv")

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Create useful date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day_of_Week"] = df["Date"].dt.dayofweek

    # Remove original Date column
    df = df.drop("Date", axis=1)

    return df


def get_preprocessor():

    numerical_features = [
        "Hour",
        "Temperature",
        "Year",
        "Month",
        "Day_of_Week"
    ]

    categorical_features = [
        "Weather",
        "Holiday",
        "Road_Type"
    ]

    numerical_transformer = SimpleImputer(
        strategy="median"
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor