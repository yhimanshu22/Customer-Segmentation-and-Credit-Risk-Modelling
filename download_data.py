import os
import subprocess
import sys

def download_datasets():
    """
    Downloads the required datasets using the Kaggle API.
    """
    # Construct path to kaggle executable in the same directory as python executable
    kaggle_path = os.path.join(os.path.dirname(sys.executable), "kaggle")

    # Dataset 1: Customer Segmentation
    # Dataset: arjunbhasin2013/ccdata
    print("Downloading Customer Segmentation dataset...")
    subprocess.run([kaggle_path, "datasets", "download", "-d", "arjunbhasin2013/ccdata", "--unzip"], check=True)
    
    # Dataset 2: Credit Risk
    # Dataset: laotse/credit-risk-dataset
    print("Downloading Credit Risk dataset...")
    subprocess.run([kaggle_path, "datasets", "download", "-d", "laotse/credit-risk-dataset", "--unzip"], check=True)

    print("Datasets downloaded successfully.")

if __name__ == "__main__":
    download_datasets()
