import os
import random
import shutil

dataset_path = "combined_dataset/train"

target_count = 500

image_ext = (".jpg",".jpeg",".png",".bmp")

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [img for img in os.listdir(class_path) if img.lower().endswith(image_ext)]

    print(f"\n{class_name} before:", len(images))

    # if folder empty
    if len(images) == 0:
        print("No images found, skipping class")
        continue

    # UNDERSAMPLING
    if len(images) > target_count:

        remove_images = random.sample(images, len(images) - target_count)

        for img in remove_images:
            os.remove(os.path.join(class_path, img))

    # OVERSAMPLING
    elif len(images) < target_count:

        while len(images) < target_count:

            img = random.choice(images)

            src = os.path.join(class_path, img)

            new_name = f"copy_{random.randint(10000,99999)}_{img}"

            dst = os.path.join(class_path, new_name)

            shutil.copy(src, dst)

            images.append(new_name)

    print(f"{class_name} after:", len(os.listdir(class_path)))

print("\nDataset balancing completed")
