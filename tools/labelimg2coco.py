import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET


"""
labelimg-xml 数据集转换为 coco 数据集
---
- root
    -- annnotations
        --- instances_train2017.json
        --- instances_val2017.json
    -- train2017
    -- val2017
    -- test2017
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


def get_and_check(root, name, length):
    """
    XML 文件获取数据
    """
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError(
            'The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def annotations_convert_json(annotation_files, annotation_classes, image_suffix='.png'):
    """
    根据标注文件，将各个数据集标注文件转换为 COCO JSON 文件 
    - annotation_files , 标注文件
    - annotation_classes , 标注目标分类信息
    - image_suffix ,  图片后缀名
    """
    coco_json = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    # 目标类别内部编号 { 'cat':0 , 'dog':1 }
    categories = {}
    for index, class_name in enumerate(annotation_classes):
        categories[str(class_name)] = index
        category_json = {'supercategory': 'none',
                         'id': index, 'name': str(class_name)}
        coco_json['categories'].append(category_json)
    # 遍历标注框信息
    box_bounding_id = 1
    for index, line in enumerate(annotation_files):
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()
        # 图片信息
        filename = os.path.basename(xml_f)[:-4] + image_suffix
        image_id = 202300000001 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': image_id
        }
        coco_json['images'].append(image)
        # 标准框信息
        objs = root.findall('object')
        for obj in objs:
            # 标记时，使用数字代替标准信息
            category = get_and_check(obj, 'name', 1).text
            # 获取类别 ID
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert (xmax > xmin), "xmax <= xmin, {}".format(line)
            assert (ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                'area': o_width*o_height,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [xmin, ymin, o_width, o_height],
                'category_id': category_id, 'id': box_bounding_id, 'ignore': 0,
                'segmentation': []
            }
            coco_json['annotations'].append(ann)
            box_bounding_id = box_bounding_id + 1
    # 返回 coco_json
    return coco_json


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


def coco_annotations_data(root_dir, classes, train_files, val_files, test_files, image_suffix=".png"):
    """
    标注文件数据：创建 coco/annotations 文件夹及其内容
    - {root_dir}/annotations/instances_train2017.json
    - {root_dir}/annotations/instances_val2017.json
    - {root_dir}/annotations/instances_test2017.json
    """
    # 创建目录
    annotations_dir = f'{root_dir}/annotations'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    if not os.path.exists(annotations_dir):
        os.mkdir(annotations_dir)
    # 训练集
    coco_train_file = f'{annotations_dir}/instances_train2017.json'
    coco_train_json = annotations_convert_json(
        annotation_files=train_files,
        annotation_classes=classes,
        image_suffix=image_suffix,
    )
    coco_train_data = json.dumps(coco_train_json)
    create_new_file(coco_train_file, coco_train_data)
    # 验证集
    coco_val_file = f'{annotations_dir}/instances_val2017.json'
    coco_val_json = annotations_convert_json(
        annotation_files=val_files,
        annotation_classes=classes,
        image_suffix=image_suffix,
    )
    coco_val_data = json.dumps(coco_val_json)
    create_new_file(coco_val_file, coco_val_data)
    # 测试集
    coco_test_file = f'{annotations_dir}/instances_test2017.json'
    coco_test_json = annotations_convert_json(
        annotation_files=test_files,
        annotation_classes=classes,
        image_suffix=image_suffix,
    )
    coco_test_data = json.dumps(coco_test_json)
    create_new_file(coco_test_file, coco_test_data)


def create_copy_image(annotation_files, image_dir, image_suffix):
    """
    复制图片数据，返回图片列表 
    """
    # 创建文件夹
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    # 复制图片
    image_infos = []
    for annotation_file in annotation_files:
        img = annotation_file[:-4] + image_suffix
        # 复制图片
        new_img = f'{image_dir}/{os.path.basename(img)}'
        shutil.copyfile(img, new_img)
        # 保存图片地址
        image_infos.append(new_img)
    return image_infos


def coco_images_data(root_dir, train_files, val_files, test_files, image_suffix=".png"):
    """
    图片数据整理：coco/train2017 ，coco/val2017 , coco/test2017  
    """
    # 根目录
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    # 训练集图片
    train_dir = f'{root_dir}/train2017'
    train_info_file = f'{root_dir}/train.txt'
    train_infos = create_copy_image(
        annotation_files=train_files,
        image_dir=train_dir,
        image_suffix=image_suffix
    )
    train_info_data = '\n'.join(train_infos)
    create_new_file(train_info_file, train_info_data)
    # 验证集数据
    val_dir = f'{root_dir}/val2017'
    val_info_file = f'{root_dir}/val.txt'
    val_infos = create_copy_image(
        annotation_files=val_files,
        image_dir=val_dir,
        image_suffix=image_suffix
    )
    val_info_data = '\n'.join(val_infos)
    create_new_file(val_info_file, val_info_data)
    # 测试集数据
    test_dir = f'{root_dir}/test2017'
    test_info_file = f'{root_dir}/test.txt'
    test_infos = create_copy_image(
        annotation_files=test_files,
        image_dir=test_dir,
        image_suffix=image_suffix
    )
    test_info_data = '\n'.join(test_infos)
    create_new_file(test_info_file, test_info_data)


def coco_classes_data(root_dir, classes):
    """ 
    创建标注类别文件 class.txt
    """
    # 根目录
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    # 文件
    classes_file = f'{root_dir}/class.txt'
    classes_data = '\n'.join(map(str,classes))
    create_new_file(classes_file, classes_data)


def labelimg2coco(labelimg_dir, coco_dir, classes, train_val_ratio=0.7, image_suffix=".png"):
    """
    labelimg XML 数据转 coco 数据集
    - train_val_ratio 测试集，验证集占比
    """
    # 标注数据
    train_files, val_files,  test_files = annotations_datasets(
        annotation_dir=labelimg_dir,
        train_val_ratio=train_val_ratio
    )
    # 标注文件
    coco_annotations_data(
        root_dir=coco_dir,
        classes=classes,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        image_suffix=image_suffix
    )
    coco_images_data(
        root_dir=coco_dir,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        image_suffix=image_suffix
    )
    # 创建
    coco_classes_data(
        root_dir=coco_dir,
        classes=classes
    )
    # 信息
    print(
        f'train size {len(train_files)} , val size {len(val_files)} , test size {len(test_files)}')


if __name__ == '__main__':
    # 标注数据转 COCO 数据集
    labelimg2coco(
        labelimg_dir='mapdata',
        coco_dir='cocomap',
        classes=[0, 1, 2]
    )
