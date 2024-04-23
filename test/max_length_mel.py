import os
import numpy as np

def load_arrays_from_folder(folder_path):
    x_values = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            array = np.load(os.path.join(folder_path, filename))
            x = array.shape[1] 
            x_values.append(x)
    return x_values

folder_path = "/home/satvik/Documents/Sem-4/mlpr/TheOnlyCAPTYouNeed/test/final_dataset/train/l1/audio"
x_values = load_arrays_from_folder(folder_path)

# Sort the x_values
x_values.sort()

# Calculate quartiles
quartiles = np.percentile(x_values, [25, 50, 95])

print("All quartiles of x:", quartiles)
