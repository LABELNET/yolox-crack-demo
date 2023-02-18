import os
import glob
import numpy as np


"""
labelimg-xml 数据集转换为 voc 数据集
---
VOCdevkit
    --VOC2007
        --Annotations : xml 标注文件
        --JPEGImages  : 图片文件
        --ImageSets   : 训练参数
            --Main
                --train.txt
                --test.txt
                --trainval.txt
                --val.txt
"""


def annotations_datasets(annotation_dir, train_val_ratio=0.7):
    """ 
    根据标注文件，随机划分数据集
    """
    # 标注文件
    annotations_files = glob.glob(annotation_dir + "/*.xml")
    # 排序
    annotations_files.sort()
    # 洗牌打乱顺序
    np.random.shuffle(annotations_files)
    # 数据集比例
    train_ratio = 1
    val_ratio = 0
    if train_val_ratio < 1:
        train_ratio = train_val_ratio
        val_ratio = 0.9-train_val_ratio
        if val_ratio < 0:
            val_ratio = 0
    # 数据集下标
    train_index, val_index = int(
        len(annotations_files)*train_ratio), int(len(annotations_files)*val_ratio)
    # 按比例划分数据集
    train_files, val_files,  test_files = annotations_files[:train_index], annotations_files[
        train_index:train_index+val_index], annotations_files[train_index+val_index:]
    return train_files, val_files,  test_files


def create_new_file(new_file, data):
    """
    创建新文件，删除和创建
    """
    if os.path.exists(new_file):
        # 若存在，先删除
        os.remove(new_file)
    with open(new_file, 'w') as file:
        file.write(data)
    # 输出
    print(new_file)


def voc_annotation_data(labelimg_dir):
    """
    创建 VOC 标注数据文件夹， VOCdevkit/VOC2007/Annotations
    """
    annotation_dir = 'VOCdevkit/VOC2007/Annotations'
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
    # 复制标注文件
    os.system(f'cp {labelimg_dir}/*.xml {annotation_dir}/')


def voc_image_data(labelimg_dir, image_suffix='.png'):
    """
    创建 VOC 图片数据文件夹，VOCdevkit/VOC2007/JPEGImages
    """
    image_dir = 'VOCdevkit/VOC2007/JPEGImages'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    # 复制图片数据
    os.system(f'cp {labelimg_dir}/*{image_suffix} {image_dir}/')


def create_txt_data(annotation_files):
    """
    生成数据集配置 txt 内容
    """
    infos = []
    for annotation_file in annotation_files:
        img_name = annotation_file[:-4].split('/')[-1]
        infos.append(img_name)
    return '\n'.join(infos)


def voc_train_data(train_val_ratio=0.7):
    """
    创建 VOC 数据集训练  txt 文件 
    """
    train_dir = 'VOCdevkit/VOC2007/ImageSets/Main'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    # 数据集
    annotation_dir = 'VOCdevkit/VOC2007/Annotations'
    train_files, val_files,  test_files = annotations_datasets(
        annotation_dir=annotation_dir,
        train_val_ratio=train_val_ratio
    )
    # 训练集
    train_file = f'{train_dir}/train.txt'
    train_data = create_txt_data(train_files)
    create_new_file(train_file, train_data)
    # 验证集
    val_file = f'{train_dir}/val.txt'
    val_data = create_txt_data(val_files)
    create_new_file(val_file, val_data)
    # 训练集+验证集
    trainval_file = f'{train_dir}/trainval.txt'
    trainval_data = create_txt_data(train_files+val_files)
    create_new_file(trainval_file, trainval_data)
    # 测试集
    test_file = f'{train_dir}/test.txt'
    test_data = create_txt_data(test_files)
    create_new_file(test_file, test_data)

def labelimg2voc(labelimg_dir,train_val_ratio=0.7,image_suffix='.png'):
    """
    labelimg 转换 VOC 数据集 
    """
    # 标注文件
    voc_annotation_data(labelimg_dir)
    # 图片文件
    voc_image_data(labelimg_dir=labelimg_dir,image_suffix=image_suffix)
    # 训练文件
    voc_train_data(train_val_ratio=train_val_ratio)

if __name__ == '__main__':
    # 测试
    labelimg2voc(labelimg_dir='mapdata')