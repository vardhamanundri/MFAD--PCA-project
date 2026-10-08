import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

print("Downloading LFW faces dataset (this may take a minute or two)...")
# Fetch people with at least 15 images to satisfy your code's MIN_IMAGES requirement
lfw = fetch_lfw_people(min_faces_per_person=15, resize=0.5)

base_dir = "pca_face_dataset"
os.makedirs(base_dir, exist_ok=True)

saved_count = 0
for i, image in enumerate(lfw.images):
    # Get the person's name and format it for a folder name
    person_name = lfw.target_names[lfw.target[i]]
    folder_name = person_name.replace(" ", "_")
    
    # Create the person's folder
    person_dir = os.path.join(base_dir, folder_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Save the image as a .jpg
    img_path = os.path.join(person_dir, f"face_{i:04d}.jpg")
    plt.imsave(img_path, image, cmap="gray")
    saved_count += 1

print(f"Success! Saved {saved_count} images across {len(lfw.target_names)} people.")
print(f"The '{base_dir}' folder is now ready for your PCA face recognition script.")