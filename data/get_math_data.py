import kagglehub

# TODO Get this code to work
# URL: https://www.kaggle.com/datasets/awsaf49/math-qsa-dataset?resource=download

# Download latest version
path = kagglehub.dataset_download("awsaf49/math-qsa-dataset")

print("Path to dataset files:", path)