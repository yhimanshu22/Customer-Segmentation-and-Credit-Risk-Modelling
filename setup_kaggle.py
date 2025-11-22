import os
import json
import sys

def setup_kaggle_config():
    """
    Sets up the kaggle.json file with the provided credentials.
    """
    # Kaggle API credentials
    # TODO: REPLACE 'YOUR_USERNAME' WITH YOUR ACTUAL KAGGLE USERNAME
    kaggle_username = "yhimanshu22" 
    kaggle_key = "KGAT_6007cb824c7fbb96accdd6a234c46642"

    if kaggle_username == "YOUR_USERNAME":
        print("Error: Please update the 'kaggle_username' variable in setup_kaggle.py with your actual Kaggle username.")
        sys.exit(1)

    # Define the directory for kaggle.json
    # On Linux/Mac it's usually ~/.kaggle/
    kaggle_dir = os.path.expanduser("~/.kaggle")
    
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
        print(f"Created directory: {kaggle_dir}")

    config_path = os.path.join(kaggle_dir, "kaggle.json")

    data = {
        "username": kaggle_username,
        "key": kaggle_key
    }

    with open(config_path, "w") as f:
        json.dump(data, f)
    
    # Set permissions to 600 (read/write by owner only) - required by Kaggle API
    os.chmod(config_path, 0o600)
    
    print(f"Successfully created {config_path} with provided credentials.")

if __name__ == "__main__":
    setup_kaggle_config()
