#src/data.py
import kagglehub, pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def load_raw() -> pd.DataFrame:
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(f"{path}/creditcard.csv")
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    df["hour"] = (df.Time//3600) % 24
    return df

def get_splits(df: pd.DataFrame):
    """Stratified 80/20. Without stratified a random split can hand you a test set holding 40% of all fraud"""
    X = df.drop(columns=["Class","hour"])
    y = df["Class"]
    return train_test_split(X, y, test_size = 0.2, stratify = y, random_state = RANDOM_STATE)