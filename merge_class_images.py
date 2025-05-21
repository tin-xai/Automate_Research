import os
import cv2
# from wand.image import Image
# from wand.color import Color
# from wand.drawing import Drawing
# from wand.font import Font

# def run_one_class(class_idx, saved_folder):
#     path = f'/home/giang/Downloads/val/{class_idx}'
#     files = os.listdir(path)
#     if not os.path.exists(f"{saved_folder}"):
#         os.mkdir(f"{saved_folder}")

#     font = Font(path='LeagueGothic_Regular.otf', size=15)
#     # concat horizontal
#     outputs = []
#     for i in range(0, len(files), 10):
#         path1 = f"{path}/{files[i]}"

#         h_images = None
#         for j in range(i+1, i+10):
#             if j >= len(files):
#                 break
#             path2 = f"{path}/{files[j]}"
            
#             if j == i + 1:
#                 left_image = Image(filename=path1)
#             else:
#                 left_image = h_images.clone()
#             right_image = Image(filename=path2)
#             if j == i + 1:
#                 left_image.resize(200, 200)
#             right_image.resize(200, 200)

#             # add lower white space to right image to add info on that space
#             clr = Color('white')
#             if j == i + 1:
#                 with Image(width = 200, height = 250, background = clr) as white_image_1:
#                     white_image_1.composite(image=left_image, left=0, top=0)
#                     left_image = white_image_1.clone()

#                 # with Drawing() as draw:
#                 #     draw.font = '/home/tin/new_factCheck/factCheck/src/merged_images_2/LeagueGothic_Regular.otf'
#                 #     draw.font_size = 15
#                 #     draw.text(10, left_image.height-10, str(files[j-1].split('.')[0]))
#                 #     draw(left_image)
#                 left_image.caption(str(files[j-1].split('.')[0]), left=10, top=left_image.height-10,
#                       font=font,
#                       gravity="center")

#             with Image(width = 200, height = 250, background = clr) as white_image_2:
#                 white_image_2.composite(image=right_image, left=0, top=0)
#                 right_image = white_image_2.clone()

#             # with Drawing() as draw:
#             #     font = Font(path='/home/tin/new_factCheck/factCheck/src/merged_images_2/LeagueGothic_Regular.otf', size=15)
#                 # draw.font = font
#                 # draw.font_size = 15
#                 # draw.text(10, right_image.height-10, str(files[j].split('.')[0]))
#                 # draw(right_image)
            
#             right_image.caption(str(files[j].split('.')[0]), left=10, top=right_image.height-10,
#                       font=font,
#                       gravity="center")
                
            
#             width=left_image.width+right_image.width
#             height=max(left_image.height, right_image.height)
            
#             with Image(width=width, height=height) as output:
#                 output.composite(image=left_image, left=0, top=0)
#                 output.composite(image=right_image, left=left_image.width, top=0)
#                 h_images = output.clone()

#         outputs.append(h_images.clone())

#     # concat vertical
#     up_image = outputs[0]
#     v_images = None
#     for i in range(1, len(outputs)):
#         down_image = outputs[i]

#         with Image(width=up_image.width, height=up_image.height + down_image.height) as output2:
#             output2.composite(image=up_image, left=0, top=0)
#             output2.composite(image=down_image, left=0, top=up_image.height)
#             v_images = output2.clone()
            
#         up_image = v_images.clone()
#     v_images.save(filename=f"./{saved_folder}/combine_{class_idx}.png")

def read_img2label():
    category_path = '/Users/tinnguyen/Downloads/XCLIP/365Places/filelist_places365-standard/categories_places365.txt'
    val_label_path = '/Users/tinnguyen/Downloads/XCLIP/365Places/filelist_places365-standard/places365_val.txt'

    idx2cat = {}
    cat2idx = {}
    img2idx = {}

    f = open(val_label_path, 'r')
    lines = f.readlines()
    for line in lines:
        img_name = line.split(' ')[0]
        label_idx = line.split(' ')[1]
        if '\n' in label_idx:
            label_idx = label_idx[:-1]

        img2idx[img_name] = label_idx
    f.close()
    
    f = open(category_path, 'r')
    lines = f.readlines()
    for line in lines:
        cat = line.split(' ')[0]
        label_idx = line.split(' ')[1]
        if '\n' in label_idx:
            label_idx = label_idx[:-1]

        idx2cat[label_idx] = cat
        cat2idx[cat] = label_idx
    f.close()
    
    return idx2cat, cat2idx, img2idx

def group_images2cats(img2idx, idx2cat):
    val_img_path = '/Users/tinnguyen/Downloads/XCLIP/365Places/val_large/'
    files = os.listdir(val_img_path)

    cat2imgs = {}
    for f in files:
        cat = idx2cat[img2idx[f]]
        if cat not in cat2imgs:
            cat2imgs[cat] = []
        cat2imgs[cat].append(val_img_path + f)
    
    for k in list(cat2imgs.keys()):
        print(len(cat2imgs[k]))
        

    
if __name__ == '__main__':
    idx2cat, cat2idx, img2idx = read_img2label()
    group_images2cats(img2idx, idx2cat)
    # run on some classes
    # class_indexes = read_idx2label()
    # saved_folder = './merged_images/'
    # for cls in class_indexes:
    #     run_one_class(cls, saved_folder)
        
