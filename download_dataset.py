
import kagglehub
import os
import shutil

# Download dataset
path = kagglehub.dataset_download(
    "abhinavkulshreshth/pothole-detection-dataset"
)

print("Dataset downloaded to:", path)

# Create project data directory
os.makedirs("data", exist_ok=True)

# Copy downloaded dataset into project
for folder in ["train", "test", "val"]:
    source = os.path.join(path, folder)
    destination = os.path.join("data", folder)

    if os.path.exists(source):
        if os.path.exists(destination):
            shutil.rmtree(destination)

        shutil.copytree(source, destination)
        print(folder, "copied successfully")
    else:
        print("Folder not found:", source)

print("Dataset preparation completed.")
