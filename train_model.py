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
    # Sample 60k rows — plenty for a good model, saves RAM during training
    df = pd.read_csv(data_path)
    if len(df) > 60000:
        df = df.sample(60000, random_state=42)
        print(f"[INFO] Sampled to 60,000 rows to fit free-tier memory")
    df = df.dropna()
    print(f"[INFO] Shape after dropna: {df.shape}")

    OR_COLS  = ['education_level', 'company_size']
    OHE_COLS = ['job_title', 'industry', 'location', 'remote_work']

    cdf = df.select_dtypes(include='object')
    ndf = df.select_dtypes(exclude='object')

    oe  = OrdinalEncoder()
    ohe = OneHotEncoder(sparse_output=False)

    oe_enc  = oe.fit_transform(cdf[OR_COLS])
    ohe_enc = ohe.fit_transform(cdf[OHE_COLS])

    oe_enc_df  = pd.DataFrame(oe_enc,  columns=OR_COLS)
    ohe_enc_df = pd.DataFrame(ohe_enc, columns=ohe.get_feature_names_out())

    for col_name, categories in zip(OR_COLS, oe.categories_):
        print(f"\nOrdinal mappings for: {col_name}")
        for idx, cls in enumerate(categories):
            print(f"  {idx} -> {cls}")

    final_cdf = pd.concat([ohe_enc_df, oe_enc_df], axis=1)
    final_df  = pd.concat([final_cdf, ndf], axis=1)

    X = final_df.drop('salary', axis=1)
    y = final_df['salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Lightweight config: fits in 512 MB, still great accuracy
    print("\n[INFO] Training Random Forest (memory-optimised)...")
    rf = RandomForestRegressor(
        n_estimators=50,       # was 200 — biggest RAM saver
        max_depth=15,          # was 20
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=1,              # was -1, single thread uses less peak RAM
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("\n-- Evaluation --")
    print(f"MAE      : {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"R2 Score : {r2_score(y_test, y_pred):.4f}")

    os.makedirs('Best_model', exist_ok=True)
    path = 'Best_model/random_forest_model.pkl'
    joblib.dump(rf, path, compress=3)   # compress=3 shrinks file ~60%
    print(f"\n[SUCCESS] Model saved to: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    train(args.data)
