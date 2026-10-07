"""
inspect_dataset.py
Performs comprehensive data inspection on dataset.csv:
- Columns, data types, nulls
- Duplicates
- Raw label values and class distribution
"""

import pandas as pd

def inspect(file_path="dataset.csv"):
    df = pd.read_csv(file_path)
    print("=" * 55)
    print("  DATASET INSPECTION SUMMARY (dataset.csv)")
    print("=" * 55)
    print(f"Total Rows:     {len(df):,}")
    print(f"Total Columns:  {len(df.columns)} -> {list(df.columns)}")
    print("\n--- Missing / Null Values ---")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        print(f"  {col:15}: {count} ({count/len(df)*100:.2f}%)")
    
    dup_urls = df.duplicated(subset=["url"]).sum()
    print(f"\n--- Duplicate URLs: {dup_urls} duplicate entries found")
    
    print("\n--- Raw Class Balance ---")
    val_counts = df["label"].value_counts(dropna=False)
    for val, count in val_counts.items():
        val_name = "<NULL/EMPTY>" if pd.isna(val) else str(val)
        print(f"  {val_name:15}: {count:5} ({count/len(df)*100:.2f}%)")
        
    print("\n--- Sample Rows (Head) ---")
    print(df.head(5).to_string())
    print("=" * 55)

if __name__ == "__main__":
    inspect()
