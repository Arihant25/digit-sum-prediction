import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Create output folder if it doesn't exist
output_folder = Path("smell")
output_folder.mkdir(exist_ok=True)

# Load data and labels
data = [np.load(f"data{i}.npy") for i in range(3)]
labels = [np.load(f"lab{i}.npy") for i in range(3)]

# Print dataset information
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
for i in range(3):
    print(f"\nDataset {i}:")
    print(f"  Data shape: {data[i].shape}")
    print(f"  Image dimensions: {data[i].shape[1:]} (individual image size)")
    print(f"  Number of images: {data[i].shape[0]}")
    print(f"  Labels shape: {labels[i].shape}")
    print(f"  Data dtype: {data[i].dtype}")
    print(f"  Labels dtype: {labels[i].dtype}")
    print(f"  Data range: [{data[i].min():.4f}, {data[i].max():.4f}]")
    print(f"  Labels range: [{labels[i].min()}, {labels[i].max()}]")
    print(f"  Labels unique values: {np.unique(labels[i])}")

print("\n" + "=" * 60)
print("RANDOM SAMPLES")
print("=" * 60)

# Select random indices and print images
for i in range(3):
    random_idx = np.random.randint(0, len(data[i]))
    image = data[i][random_idx]
    label = labels[i][random_idx]
    
    print(f"\nDataset {i}: Random image index {random_idx}, Label: {label}")
    
    # Save the image
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f"Dataset {i} - Label: {label}")
    plt.axis('off')
    plt.savefig(output_folder / f"sample_{i}.png", bbox_inches='tight', dpi=100)
    plt.close()
    
    print(f"  Saved to smell/sample_{i}.png")

print("\nDone! 3 random samples saved to the 'smell' folder.")
