import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle 

def preprocess_data(raw_data_path: str, output_dir: str):
    """
    Melakukan data loading, cleaning, dan preprocessing pada dataset.

    Args:
        raw_data_path (str): Path ke file CSV data mentah.
        output_dir (str): Path folder untuk menyimpan data hasil preprocessing dan transformer.
    
    Returns:
        tuple: (X_train_processed_df, X_test_processed_df, y_train, y_test)
    """
    print(f"Memuat data dari: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Menangani Missing Values dan Duplikat
    df_cleaned = df.dropna().copy()
    df_cleaned.drop_duplicates(inplace=True)
    
    print(f"Data bersih memiliki shape: {df_cleaned.shape}")

    # Pemisahan Fitur dan Target
    X = df_cleaned.drop('Salary', axis=1)
    y = df_cleaned['Salary']
    
    # Pembagian Data (Training dan Testing) 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Definisi Transformasi
    numerical_features = ['Rating', 'Salaries Reported']
    categorical_features = ['Company Name', 'Job Title', 'Location'] 

    # Membuat ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Fitting dan Transforming
    print("Memulai fitting dan transforming data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Konversi hasil transformasi (sparse matrix/array) kembali ke DataFrame
    feature_names = preprocessor.get_feature_names_out()

    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    
    # Penyimpanan Artefak Preprocessing (penting wak)
    # Simpan preprocessor (ColumnTransformer) supaya transformasi nya bisa diulang
    with open(f'{output_dir}/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Simpan data hasil preprocessing (CSV)
    pd.concat([X_train_processed_df, y_train], axis=1).to_csv(
        f'{output_dir}/train_processed.csv', index=False
    )
    pd.concat([X_test_processed_df, y_test], axis=1).to_csv(
        f'{output_dir}/test_processed.csv', index=False
    )

    print(f"Data preprocessing selesai. Artefak disimpan di {output_dir}")
    
    return X_train_processed_df, X_test_processed_df, y_train, y_test

if __name__ == '__main__':
    RAW_DATA_PATH = 'https://raw.githubusercontent.com/Nuno-Hadianto/Eksperimen_SML_Mohammed-Noeno-Hadianto/refs/heads/main/Software_Professional_Salaries.csv' 
    OUTPUT_DIR = 'preprocessing/namadataset_preprocessing' 
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Jalankan fungsi
    preprocess_data(RAW_DATA_PATH, OUTPUT_DIR)