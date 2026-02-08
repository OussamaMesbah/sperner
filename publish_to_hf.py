import os
from huggingface_hub import HfApi, login

def publish(token=None, username="omesbah"):
    api = HfApi()
    
    if token:
        login(token=token)
    
    info = api.whoami()
    username = info['name']
    print(f"Authenticated as {username}")
    
    # --- 1. MODEL REPOSITORY (Code & Library) ---
    model_id = f"{username}/topo-align"
    print(f"\n[1/3] Preparing Model Repo: {model_id}")
    try:
        api.create_repo(repo_id=model_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=".",
            repo_id=model_id,
            repo_type="model",
            ignore_patterns=[".venv/*", "__pycache__/*", ".git/*", "dist/*", "sperner_dataset.json", "*.pyc", "equilib_login.py"]
        )
        print("Model Repo upload complete.")
    except Exception as e:
        print(f"Error updating model repo: {e}")

    # --- 2. DATASET REPOSITORY (Sperner-Bench) ---
    dataset_id = f"{username}/sperner-bench"
    print(f"\n[2/3] Preparing Dataset Repo: {dataset_id}")
    try:
        api.create_repo(repo_id=dataset_id, repo_type="dataset", exist_ok=True)
        # Create a simple dataset README
        with open("DATASET_README.md", "w", encoding="utf-8") as f:
            f.write("---\ntask_categories:\n- reasoning\n- graph-learning\nlicense: mit\n---\n")
            f.write("# Sperner-Bench\n\nHigh-dimensional dataset for benchmarking topological alignment solvers.\n")
            f.write("Includes ground truth labels for objective preference boundaries.")
        
        api.upload_file(
            path_or_fileobj="DATASET_README.md",
            path_in_repo="README.md",
            repo_id=dataset_id,
            repo_type="dataset"
        )
        api.upload_file(
            path_or_fileobj="sperner_dataset.json",
            path_in_repo="sperner_dataset.json",
            repo_id=dataset_id,
            repo_type="dataset"
        )
        print("Dataset Repo upload complete.")
    except Exception as e:
        print(f"Error updating dataset repo: {e}")

    # --- 3. SPACE REPOSITORY (Interactive Demo) ---
    space_id = f"{username}/topo-align-demo"
    print(f"\n[3/3] Preparing Space Repo: {space_id}")
    try:
        # Note: 'sdk' is correct for newer hub versions, but handled by create_repo
        api.create_repo(repo_id=space_id, repo_type="space", sdk="streamlit", exist_ok=True)
        # Upload app.py, requirements.txt and the lib
        api.upload_folder(
            folder_path=".",
            repo_id=space_id,
            repo_type="space",
            ignore_patterns=[".venv/*", "__pycache__/*", ".git/*", "dist/*", "sperner_dataset.json", "topo_align_showcase.ipynb", "DATASET_README.md"]
        )
        print("Space Demo upload complete.")
    except Exception as e:
        print(f"Error updating space: {e}")
    
    print("\n" + "="*40)
    print(" ECOSYSTEM PUBLISHED SUCCESSFULLY")
    print("="*40)
    print(f"Library: https://huggingface.co/models/{model_id}")
    print(f"Dataset: https://huggingface.co/datasets/{dataset_id}")
    print(f"Demo:    https://huggingface.co/spaces/{space_id}")

if __name__ == "__main__":
    import sys
    token = sys.argv[1] if len(sys.argv) > 1 else None
    publish(token)
