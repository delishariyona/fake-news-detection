from src.data_loader import load_data
from src.preprocessing import clean_text
from src.train_models import train_all_models
from src.evaluate import compare_models
import pandas as pd

def main():
    print("📰 Fake News Detection Pipeline Started")

    
    df = load_data("data/archive (10).zip")
    df["combined_text"] = df["title"] + " " + df["text"]
    df["cleaned_text"] = df["combined_text"].apply(clean_text)
    print(f"✅ Data loaded and cleaned: {df.shape[0]} samples")

    
    results, tfidf, models, test_data = train_all_models(df)
    print("✅ Models trained successfully!")

    
    compare_models(results)

    
    pd.DataFrame(results).T.round(4).to_csv("outputs/reports.txt")
    print("📁 Results saved in 'outputs/' folder")

if __name__ == "__main__":
    main() 