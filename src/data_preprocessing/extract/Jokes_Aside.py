import kagglehub
import pandas as pd
# Download latest version
path = kagglehub.dataset_download("fabiodeponte/the-complete-humour-project-jokes-aside")

print("Path to dataset files:", path)

