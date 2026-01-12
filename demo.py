import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from palms import affine_linear_partitioning

def run_demo():
    img_path = 'redMacaw.jpg'
    try:
        img_pil = Image.open(img_path)
        img = np.array(img_pil).astype(np.float64) / 255.0
    except FileNotFoundError:
        print(f"Image {img_path} not found. Please provide an image.")
        return

    print(f"Running PALMS on image of shape {img.shape}...")
    
    gamma = 0.4 
    u, partition, a, b, c = affine_linear_partitioning(img, gamma=gamma, verbose=True)

    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(img)
    ax[0].set_title("Input")
    ax[0].axis('off')
    
    ax[1].imshow(np.clip(u, 0, 1))
    ax[1].set_title("Piecewise Affine Approx")
    ax[1].axis('off')
    
    ax[2].imshow(partition, cmap='nipy_spectral')
    ax[2].set_title("Partition")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()