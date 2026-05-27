"""
train_model.py — memory-optimised for Render free tier (512 MB)
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(data_path: str) -> None:
    print(f"[INFO] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    if len(df) > 60000:
        df = df.sample(60000, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled to 60,000 rows")

    # Drop any rows with missing values
    df = df.dropna().reset_index(drop=True)
    print(f"[INFO] Shape after dropna: {df.shape}")

    OR_COLS  = ['education_level', 'company_size']
    OHE_COLS = ['job_title', 'industry', 'location', 'remote_work']
    NUM_COLS = ['experience_years', 'skills_count', 'certifications']
    TARGET   = 'salary'

    # Explicitly cast types to avoid pandas 3.0 str/object issues
    for col in OR_COLS + OHE_COLS:
        df[col] = df[col].astype(str)
    for col in NUM_COLS + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)
    print(f"[INFO] Shape after type cast + dropna: {df.shape}")

    oe  = OrdinalEncoder()
    ohe = OneHotEncoder(sparse_output=False)

    oe_enc  = oe.fit_transform(df[OR_COLS])
    ohe_enc = ohe.fit_transform(df[OHE_COLS])

    oe_enc_df  = pd.DataFrame(oe_enc,  columns=OR_COLS)
    ohe_enc_df = pd.DataFrame(ohe_enc, columns=ohe.get_feature_names_out())

    for col_name, categories in zip(OR_COLS, oe.categories_):
        print(f"\nOrdinal mappings for: {col_name}")
        for idx, cls in enumerate(categories):
            print(f"  {idx} -> {cls}")

    X = pd.concat([ohe_enc_df, oe_enc_df, df[NUM_COLS].reset_index(drop=True)], axis=1)
    y = df[TARGET].reset_index(drop=True)

    print(f"\n[INFO] NaN in X: {X.isna().sum().sum()}, NaN in y: {y.isna().sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n[INFO] Training Random Forest (memory-optimised)...")
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("\n-- Evaluation --")
    print(f"MAE      : {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"R2 Score : {r2_score(y_test, y_pred):.4f}")

    os.makedirs('Best_model', exist_ok=True)
    path = 'Best_model/random_forest_model.pkl'
    joblib.dump(rf, path, compress=3)
    print(f"\n[SUCCESS] Model saved to: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    train(args.data)
