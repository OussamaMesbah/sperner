import os
import streamlit as st

# Import the main UI logic
# Since the Space environment might have its own directory structure, 
# we ensure the project root is in the path.
import sys
sys.path.append(os.path.dirname(__file__))

# Run the UI
from equilib.human_ui import main

if __name__ == "__main__":
    main()
