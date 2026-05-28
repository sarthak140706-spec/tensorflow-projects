import glob
from sklearn.model_selection import train_test_split

def load_data(data_dir, test_size=0.2, val_size=0.1, random_state=42):

    images = []
    labels = []

    for class_folder in ["with_mask","without_mask"]:
        for image_file in glob.glob(data_dir + "/" + class_folder + "/*"):
            images.append(image_file)
            labels.append(class_folder)  

    train_images, temp_images, train_labels, temp_labels =train_test_split(images, labels, test_size=(test_size + val_size), random_state=random_state)
    
    val_ratio = val_size / (val_size + test_size)  # proportion of val in temp
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=(1 - val_ratio), random_state=random_state
    )

    return train_images, train_labels, val_images, val_labels, test_images, test_labels