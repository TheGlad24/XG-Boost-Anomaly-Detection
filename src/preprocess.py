import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

RAW_DIR = "data/raw/"
SAVE_PATH = "data/processed/combined_cleaned.csv"

# List of column names for UNSW-NB15 (49 total)
UNSW_COLS = [
    'id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
    'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
    'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
    'attack_cat', 'label', 'srcip', 'sport', 'dstip', 'dsport'
]

def load_and_combine_unsw_csvs():
    all_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    df_list = []
    for f in sorted(all_files):
        path = os.path.join(RAW_DIR, f)
        df = pd.read_csv(path, header=None, names=UNSW_COLS, low_memory=False)
        print(f"[INFO] Loaded {f} with shape {df.shape}")
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def clean_and_encode(df):
    # Drop non-usable columns
    df = df.drop(['id', 'srcip', 'dstip', 'sport', 'dsport'], axis=1, errors='ignore')

    # Fill missing attack_cat
    df['attack_cat'] = df['attack_cat'].fillna('Normal')

    # Encode target class
    df['attack_cat'] = LabelEncoder().fit_transform(df['attack_cat'].astype(str))

    # Encode all remaining object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Drop 'label' (binary) since we're using multiclass attack_cat
    df = df.drop(['label'], axis=1, errors='ignore')

    # Normalize numeric features
    feature_cols = df.drop(['attack_cat'], axis=1).columns
    df[feature_cols] = MinMaxScaler().fit_transform(df[feature_cols])

    return df

def main():
    print("Loading and combining CSV files...")
    df = load_and_combine_unsw_csvs()
    print("Cleaning and encoding...")
    df_cleaned = clean_and_encode(df)
    print(f"Saving to {SAVE_PATH}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df_cleaned.to_csv(SAVE_PATH, index=False)

if __name__ == "__main__":
    main()
