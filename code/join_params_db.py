import os
import shutil

n = input("Enter the number: ")
dest_folder = os.path.join("DB", "ACCIDENTS", f"accident{n}")

src_file = "cars_parameters.pt"  # Assuming it's in the current directory
dest_file = os.path.join(dest_folder, f"cars_parameters{n}.pt")

# Move the file
shutil.move(src_file, dest_file)

print(f"Moved to: {dest_file}")
