import pandas as pd
import argparse
import os

def preprocess_data(input_path, output_path):
    """
    Memuat dataset mentah, melakukan preprocessing (menangani missing values),
    dan menyimpan dataset yang sudah bersih.
    """
    print(f"Memuat dataset dari: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {input_path}")
        return
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return

    print("Memulai preprocessing data...")

    # Strategi 1: Mengisi nilai kosong dengan "Unknown"
    df['director'].fillna('Unknown', inplace=True)
    df['cast'].fillna('Unknown', inplace=True)
    df['country'].fillna('Unknown', inplace=True)

    # Strategi 2: Menghapus baris yang memiliki nilai kosong pada kolom tertentu
    df.dropna(subset=['date_added', 'rating', 'duration'], inplace=True)

    print("Preprocessing selesai. Tidak ada missing values.")

    # Memastikan folder output ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Menyimpan dataset yang sudah bersih
    try:
        df.to_csv(output_path, index=False)
        print(f"Dataset bersih telah disimpan di: {output_path}")
    except Exception as e:
        print(f"Error saat menyimpan data: {e}")

if __name__ == "__main__":
    # Ini membuat script bisa dijalankan dari command line
    parser = argparse.ArgumentParser(description="Script otomatisasi preprocessing data Netflix.")
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="Path ke file dataset mentah (misal: ../namadataset_raw/netflix_titles.csv)"
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help="Path untuk menyimpan file CSV yang sudah bersih (misal: ../namadataset_preprocessing/netflix_preprocessed.csv)"
    )
    
    args = parser.parse_args()
    
    # Menjalankan fungsi preprocessing
    preprocess_data(args.input, args.output)