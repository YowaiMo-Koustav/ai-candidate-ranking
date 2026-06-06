import pandas as pd
import sys
import yaml

def validate_submission(csv_path: str, meta_path: str):
    print(f"Validating {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        required_cols = {'candidate_id', 'job_id', 'score'}
        if not required_cols.issubset(set(df.columns)):
            print(f"Error: Missing required columns. Expected at least: {required_cols}")
            return False
            
        print("Checking metadata...")
        with open(meta_path, 'r') as f:
            meta = yaml.safe_load(f)
            
        if not meta.get('team_name'):
            print("Error: team_name is missing in metadata.")
            return False
            
        print("Validation successful!")
        return True
    except Exception as e:
        print(f"Validation failed with error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python validate_submission.py <submission_csv> <metadata_yaml>")
        sys.exit(1)
        
    success = validate_submission(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
