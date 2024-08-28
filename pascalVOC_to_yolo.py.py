#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import xml.etree.ElementTree as ET
CLASSES = ["person", "hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]

CLASS_MAPPING = {name: idx for idx, name in enumerate(CLASSES)}
base_dir = "C:\\Users\\Kavi priya\\Desktop\\datasets"
input_images_dir = os.path.join(base_dir, "images")
input_annotations_dir = os.path.join(base_dir, "labels")
output_dir = os.path.join(base_dir, "yolo_format")

os.makedirs(output_dir, exist_ok=True)


# In[2]:

def convert_voc_to_yolo(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)

    yolo_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        
        if class_name not in CLASS_MAPPING:
            continue
        
        class_id = CLASS_MAPPING[class_name]

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Normalize 
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # Save 
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + ".txt")
    with open(output_file, 'w') as f:
        f.write("\n".join(yolo_labels))

    print(f"Converted {xml_file} to YOLO format and saved to {output_file}")


for xml_file in os.listdir(input_annotations_dir):
    if xml_file.endswith('.xml'):
        convert_voc_to_yolo(os.path.join(input_annotations_dir, xml_file), output_dir)


# In[3]:


import os

base_dir = "C:\\Users\\Kavi priya\\Desktop\\datasets"
images_dir = os.path.join(base_dir, "images")


print(f"Images directory path: {images_dir}")
print("Contents of the images directory:", os.listdir(images_dir))


# In[4]:


import os


files = os.listdir(images_dir)
print("Files and their extensions in the images directory:")
for file in files:
    print(file, os.path.splitext(file)[1])


# In[5]:


import os
import random
import shutil


base_dir = "C:\\Users\\Kavi priya\\Desktop\\datasets"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "yolo_format")


train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1


split_dirs = {
    "train": {"images": os.path.join(base_dir, "train", "images"), "labels": os.path.join(base_dir, "train", "labels")},
    "val": {"images": os.path.join(base_dir, "val", "images"), "labels": os.path.join(base_dir, "val", "labels")},
    "test": {"images": os.path.join(base_dir, "test", "images"), "labels": os.path.join(base_dir, "test", "labels")}
}

for split, dirs in split_dirs.items():
    os.makedirs(dirs["images"], exist_ok=True)
    os.makedirs(dirs["labels"], exist_ok=True)


all_images = []
for root, _, files in os.walk(images_dir):
    for file in files:
        if os.path.isfile(os.path.join(root, file)):
            all_images.append(os.path.relpath(os.path.join(root, file), images_dir))

print(f"Found {len(all_images)} images in the dataset.")  # Debugging: Print number of images found


random.shuffle(all_images)


train_idx = int(train_ratio * len(all_images))
val_idx = train_idx + int(val_ratio * len(all_images))


train_images = all_images[:train_idx]
val_images = all_images[train_idx:val_idx]
test_images = all_images[val_idx:]


def move_files(image_files, split):
    for image_file in image_files:
        src_image_path = os.path.join(images_dir, image_file)
        dest_image_path = os.path.join(split_dirs[split]["images"], os.path.basename(image_file))
        
        
        if os.path.isfile(src_image_path):
            print(f"Copying {src_image_path} to {dest_image_path}")  # Debugging: Print paths being copied
            shutil.copy(src_image_path, dest_image_path)

            
            label_file = os.path.splitext(image_file)[0] + ".txt"
            src_label_path = os.path.join(labels_dir, os.path.basename(label_file))
            if os.path.exists(src_label_path):
                dest_label_path = os.path.join(split_dirs[split]["labels"], os.path.basename(label_file))
                print(f"Copying label {src_label_path} to {dest_label_path}")  # Debugging: Print paths being copied
                shutil.copy(src_label_path, dest_label_path)
            else:
                print(f"Label file {label_file} not found for image {image_file}")
        else:
            print(f"Skipping non-file {src_image_path}")


move_files(train_images, "train")
move_files(val_images, "val")
move_files(test_images, "test")

print("Dataset split and moved successfully!")


# In[6]:


import os


train_images_path = r"C:\Users\Kavi priya\Desktop\datasets\train\images"
val_images_path = r"C:\Users\Kavi priya\Desktop\datasets\val\images"

print("Train images directory exists:", os.path.exists(train_images_path))
print("Validation images directory exists:", os.path.exists(val_images_path))


# In[7]:


import torch

if torch.cuda.is_available():
    print(torch.cuda.memory_summary(device=torch.device('cuda')))

