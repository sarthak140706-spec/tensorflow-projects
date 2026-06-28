"""
Data Loader for Face Mask Detection Project
"""

import os
import glob
from sklearn.model_selection import train_test_split


def load_data(
    data_dir,
    test_size=0.2,
    val_size=0.1,
    random_state=42
):
    """
    Loads image paths and labels from the dataset.

    Expected Folder Structure
    -------------------------
    data/
        with_mask/
            image1.jpg
            ...
        without_mask/
            image1.jpg
            ...

    Returns
    -------
    train_images
    train_labels
    val_images
    val_labels
    test_images
    test_labels
    """

    # Check dataset directory
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}"
        )

    images = []
    labels = []

    classes = [
        "with_mask",
        "without_mask"
    ]

    # Load image paths
    for class_name in classes:

        class_path = os.path.join(data_dir, class_name)

        if not os.path.exists(class_path):
            raise FileNotFoundError(
                f"Missing folder: {class_path}"
            )

        image_files = glob.glob(os.path.join(class_path, "*"))

        for image_file in image_files:
            images.append(image_file)
            labels.append(class_name)

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)
    print(f"Total Images : {len(images)}")
    print(f"With Mask    : {labels.count('with_mask')}")
    print(f"Without Mask : {labels.count('without_mask')}")
    print("=" * 50)

    # First split (Train + Temp)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images,
        labels,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=labels,
        shuffle=True
    )

    # Validation ratio from remaining data
    val_ratio = val_size / (test_size + val_size)

    # Second split (Validation + Test)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images,
        temp_labels,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=temp_labels,
        shuffle=True
    )

    print("Dataset Split")
    print("-" * 50)
    print(f"Training Images   : {len(train_images)}")
    print(f"Validation Images : {len(val_images)}")
    print(f"Testing Images    : {len(test_images)}")
    print("=" * 50)

    return (
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels
    )