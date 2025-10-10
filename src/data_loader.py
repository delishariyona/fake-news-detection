import pandas as pd
import zipfile
import os

def load_data(zip_path):
    """
    Loads Fake.csv and True.csv from a zip file and returns a combined dataframe with labels.
    label: 0 = Fake, 1 = Real
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} not found!")

    with zipfile.ZipFile(zip_path, 'r') as z:
        
        file_list = z.namelist()
        print("📂 Files inside ZIP:", file_list)

        
        true_file_name = next((f for f in file_list if 'True.csv' in f), None)
        fake_file_name = next((f for f in file_list if 'Fake.csv' in f), None)

        if not true_file_name or not fake_file_name:
            raise ValueError("Could not find True.csv or Fake.csv in the zip file!")

        #
        with z.open(true_file_name) as true_file:
            df_true = pd.read_csv(true_file)
        with z.open(fake_file_name) as fake_file:
            df_fake = pd.read_csv(fake_file)

    
    df_true['label'] = 1
    df_fake['label'] = 0

    
    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df
