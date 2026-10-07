import os
import glob
import shutil
import kagglehub

path = kagglehub.dataset_download("ashharfarooqui/phising-urls")
print("Downloaded to:", path)

csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
print("CSV files found:", csvs)

here = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(here, "data")
os.makedirs(data_dir, exist_ok=True)
shutil.copy(csvs[0], os.path.join(data_dir, "phishing.csv"))
print("Copied to", os.path.join(data_dir, "phishing.csv"))