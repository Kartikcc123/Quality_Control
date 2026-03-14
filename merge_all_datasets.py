import os
import shutil

tilda_path = "dataset/train"
kaggle_path = "kaggle_dataset"

combined_path = "combined_dataset/train"

mapping = {
    "good": "good",
    "hole": "hole",
    "stain": "stain",
    "horizontal": "missing_weft",
    "vertical": "missing_warp",
    "captured": "good",
    "lines": "thick_thin"
}

final_classes = [
    "good",
    "hole",
    "stain",
    "missing_warp",
    "missing_weft",
    "thick_thin"
]

# create folders
for cls in final_classes:
    os.makedirs(os.path.join(combined_path, cls), exist_ok=True)


def copy_recursive(root_folder):

    for root, dirs, files in os.walk(root_folder):

        folder_name = os.path.basename(root).lower()

        target_class = mapping.get(folder_name)

        if target_class is None:
            continue

        dst_path = os.path.join(combined_path, target_class)

        for file in files:

            if not file.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                continue

            src_file = os.path.join(root, file)

            new_name = folder_name + "_" + file

            dst_file = os.path.join(dst_path, new_name)

            shutil.copy(src_file, dst_file)

        print(f"{folder_name} → {target_class} copied")


# merge TILDA
copy_recursive(tilda_path)

# merge Kaggle
copy_recursive(kaggle_path)

print("Datasets merged successfully.")
