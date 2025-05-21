# %%
import os
import cv2
import numpy as np
import math
import pandas as pd
import json
# %%
def merge_images(image_paths, class_name="EMPTY"):
    """
    image_paths: paths of images that need to be concatenated
    class_name: class of input images
    return np image
    """

    images = []
    image_names = []
    H, W = 224, 224
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (W, H))
        images.append(image)
            
        # get file name
        image_name = image_path.split('/')[-1]
        # TODO: Append image name below the image

    len_images = len(images)
    columns, rows = 5, math.ceil(len_images/5) # predefined number of columns is 5

    extended_images = []
    for i in range(rows):
        row_images = []
        for j in range(columns):
            mapped_idx = 5*i + j
            if mapped_idx < len_images:
                row_images.append(images[mapped_idx])
            else:
                row_images.append(np.zeros((H, W, 3)))
        
        row_images = np.array(row_images)
        row_image = np.concatenate(row_images, axis=1)
        extended_images.append(row_image)
    
    extended_images = np.array(extended_images)
    extended_image = np.concatenate(extended_images, axis=0)
    
    return extended_image

# %%
def read_nabirds(data_path='/home/tin/datasets/nabirds'):
    """
    read nabird id_class dict and train_test_split dict
    """
    classes_path = os.path.join(data_path, 'classes.txt')
    train_test_path = os.path.join(data_path, 'train_test_split.txt')

    id_class_df = pd.read_table(classes_path, header=None)
    id_class_df['id'] = id_class_df[0].apply(lambda s: int(s.split(' ')[0]))
    id_class_df['class_name'] = id_class_df[0].apply(lambda s: ' '.join(s.split(' ')[1:]))
    id_class_df.drop(0, inplace=True, axis=1)

    train_test_img_df = pd.read_table(train_test_path, sep=' ', header=None)
    train_test_img_df.columns = ['image_name', 'train']
    train_test_img_df['image_name'] = train_test_img_df['image_name'].apply(lambda s: s.replace('-', ''))

    # convert df to dict
    id_class_dict = dict(zip(id_class_df['id'], id_class_df['class_name']))
    train_test_img_dict = dict(zip(train_test_img_df['image_name'], train_test_img_df['train']))

    # read valid images
    val_img_names = [k + '.jpg' for k,v in train_test_img_dict.items() if v == 0]

    return id_class_dict, train_test_img_dict, val_img_names

def read_inat21(data_path='/home/tin/datasets/inaturalist2021_onlybird'):
    """
    read nabird id_class dict and train_test_split dict
    """
    classes_path = os.path.join(data_path, 'bird_classes.json')
    train_test_path = os.path.join(data_path, 'bird_annotations.json')
    image_path = f'{data_path}/bird_train'

    classes_file = f'{data_path}/bird_classes.json'
    f = open(classes_file, 'r')
    data = json.load(f)
    id2class = data['name']
    id2imageFolder = data['image_dir_name']
    imageFolder2class = {v:id2class[k] for k, v in id2imageFolder.items()}

    # read valid images (there is no valid set in INat21 for Only Birds)
    image_folders = os.listdir(image_path)
    val_img_names = []
    for image_folder in image_folders:
        image_names = os.listdir(f'{data_path}/bird_train/{image_folder}')
        val_img_names += image_names[:25] # take 25 images per class
    
    return id2class, imageFolder2class, val_img_names
# %%
def merge_dataset(data_name='nabirds'):
    if data_name == 'nabirds':
            data_path = '/home/tin/datasets/nabirds'
    if data_name == 'inat21':
            data_path = '/home/tin/datasets/inaturalist2021_onlybird'
    if data_name == 'cub':
            data_path = '/home/tin/datasets/CUB_200_2011'

    if data_name == 'nabirds':
        save_img_folder = './merged_nabirds'
        if not os.path.exists(save_img_folder):
            os.makedirs(save_img_folder)

        # read the id-2-classname dict, imagename-2-train dict, and valid imgs
        id_class_dict, train_test_img_dict, val_img_names = read_nabirds(data_path)
    
        # read image in image folders
        img_folders = os.listdir(os.path.join(data_path, 'images'))
        for img_folder in img_folders:
            class_id = int(img_folder)
            img_names = os.listdir(os.path.join(data_path, f'images/{img_folder}'))
            # filter images, only get valid images
            img_names = [p for p in img_names if p in val_img_names]
            img_paths = [f'{data_path}/images/{img_folder}/{p}' for p in img_names]
            # run to merge images
            merged_img = merge_images(img_paths, id_class_dict[class_id])
            cv2.imwrite(f"{save_img_folder}/{class_id}_{id_class_dict[class_id]}.jpg", merged_img)
            
    elif data_name == 'inat21':
        save_img_folder = './merged_inat21'
        if not os.path.exists(save_img_folder):
            os.makedirs(save_img_folder)

        id2class, imageFolder2class, val_img_names = read_inat21(data_path)
        image_folders = os.listdir(f'{data_path}/bird_train')
        for image_folder in image_folders:
            img_names = os.listdir(f'{data_path}/bird_train/{image_folder}')
            # filter out training images
            img_names = [p for p in img_names if p in val_img_names]
            img_paths = [f'{data_path}/bird_train/{image_folder}/{p}' for p in img_names]
            # run to merge images
            merged_img = merge_images(img_paths, imageFolder2class[image_folder])
            image_folder_id = image_folder.split('_')[0]
            cv2.imwrite(f"{save_img_folder}/{image_folder_id}_{imageFolder2class[image_folder]}.jpg", merged_img)

# %%
merge_dataset(data_name='inat21')


    

        
# %%
