import os
from huggingface_hub import HfApi, create_repo, login

def publish(token=None, username="omesbah"):
    api = HfApi()
    
    # 1. Login if token provided, otherwise assume existing login
    if token:
        login(token=token)
    
    info = api.whoami()
    print(f"Authenticated as {info['name']}")
    
    # 2. Create Model Repository (for code and dataset)
    model_id = f"{username}/topo-align"
    try:
        create_repo(repo_id=model_id, repo_type="model", exist_ok=True)
        print(f"Repository {model_id} created/verified.")
    except Exception as e:
        print(f"Error creating model repo: {e}")
    
    # 3. Create Space (for interactive demo)
    space_id = f"{username}/topo-align-demo"
    try:
        create_repo(repo_id=space_id, repo_type="space", space_sdk="streamlit", exist_ok=True)
        print(f"Space {space_id} created/verified.")
    except Exception as e:
        print(f"Error creating space: {e}")
        
    # 4. Upload files to Model Repo
    print("Uploading files to model repository...")
    api.upload_folder(
        folder_path=".",
        repo_id=model_id,
        repo_type="model",
        ignore_patterns=[".venv/*", "__pycache__/*", ".git/*", "dist/*"]
    )
    
    # 5. Upload files to Space (Demo)
    print("Uploading files to Space...")
    api.upload_folder(
        folder_path=".",
        repo_id=space_id,
        repo_type="space",
        ignore_patterns=[".venv/*", "__pycache__/*", ".git/*", "dist/*", "sperner_dataset.json"] # Space might not need huge dataset if mock mode is default
    )
    
    print("\n--- Publication Complete ---")
    print(f"Model Repo: https://huggingface.co/models/{model_id}")
    print(f"Space Demo: https://huggingface.co/spaces/{space_id}")

if __name__ == "__main__":
    import sys
    # Extract token if provided, else None
    token = sys.argv[1] if len(sys.argv) > 1 else None
    publish(token)
