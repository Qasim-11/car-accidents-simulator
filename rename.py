import os


path = "DB\\ACCIDENTS"
i = 1
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    os.rename(os.path.join(path, folder), os.path.join(path, f"accident{i}"))
    i+=1
