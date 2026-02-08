from huggingface_hub import login
import getpass
import sys

def secure_login():
    print("--- Topo-Align Hugging Face Authentication ---")
    print("Please generate a 'Write' token at: https://huggingface.co/settings/tokens")
    try:
        token = getpass.getpass("Paste your Hugging Face Write Token (input is hidden): ")
        if not token:
            print("Error: No token provided.")
            return
        
        login(token=token, add_to_git_credential=True)
        print("\n[SUCCESS] Login successful! Your session is now saved.")
        print("You can now tell the assistant: 'I am logged in'.")
    except Exception as e:
        print(f"\n[ERROR] Login failed: {e}")
        print("Please ensure you have an active internet connection and the token is valid.")

if __name__ == "__main__":
    secure_login()
