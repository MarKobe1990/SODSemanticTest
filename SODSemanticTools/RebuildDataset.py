import datetime

import cv2
import numpy as np
from tensorboardX import SummaryWriter
from myTools.data import GetDataset
from mmcls.apis import init_model, inference_model, show_result_pyplot
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms
import myTools.imageUtil as utils
import os.path as osp
import os


def init_my_model(config_file, checkpoint_file, device):
    # 配置文件
    instance_model = init_model(config_file, checkpoint_file, device=device)
    return instance_model


def rebuild_dataset(dataset_dic, model):
    data_set = GetDataset(dataset_dic.get("image_root"), dataset_dic.get("depth_root"), dataset_dic.get("gt_root"))
    pbar = tqdm(range(len(data_set)), desc="rebuild_dataset:", unit='img')
    date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    save_dir = "../testResults/" + date_str
    dataset_writer = SummaryWriter(save_dir + "/log", comment="log")
    f = open("../myDocument/imageNet1k-class.txt")
    class_list = f.readlines()
    for idx in pbar:
        name, image_path, depth_path, gt_path, img_array = data_set.load_data(idx)
        result = inference_model(model, img_array)
        for class_info in class_list:
            class_info_list = class_info.split(" ")
            if str(result.get("pred_label")) == class_info_list[0]:
                result["pred_hypernyms_class"] = class_info_list[2].strip(",")
        hparam_dic = {
            "file_pth": image_path,
            "pred_class": result.get("pred_class"),
            "pred_hypernyms_class": result.get("pred_hypernyms_class"),
        }
        metric_dic = {
            "pred_label": result.get("pred_label"),
            "pred_score": result.get("pred_score")
        }
        dataset_writer.add_hparams(hparam_dic, metric_dic, name='log/' + image_path)
        # dataset_writer.add_image()
        # show_result_pyplot(model, image_path, result)


def draw_max_rec_of_mask(gt_path, image_path):
    gt_mask = cv2.imread(gt_path)
    thresh = cv2.Canny(gt_mask, 128, 256)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_origin = cv2.imread(image_path)
    img_copy = np.copy(img_origin)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for contour in contours:
        # 找到边界坐标
        x, y, w, h = cv2.boundingRect(contour)  # 计算点集最外面的矩形边界
        # 因为这里面包含了，图像本身那个最大的框，所以用了if，来剔除那个图像本身的值。
        # if x != 0 and y != 0 and w != gt_mask.shape[1] and h != gt_mask.shape[0]:
        # if w != gt_mask.shape[1] and h != gt_mask.shape[0]:
        # 左上角坐标和右下角坐标
        # 如果执行里面的这个画框，就是分别来画的，
        # cv2.rectangle(origin_gt, (x, y), (x + w, y + h), (0, 255, 0), 1)
        x1.append(x)
        y1.append(y)
        x2.append(x + w)
        y2.append(y + h)
    x11 = min(x1)
    y11 = min(y1)
    x22 = max(x2)
    y22 = max(y2)
    white = [255, 255, 255]
    for col in range(x11, x22):
        for row in range(y11, y22):
            gt_mask[row, col] = white
    rectangle_mask_gray = cv2.cvtColor(gt_mask, cv2.COLOR_RGB2GRAY)
    out_origin_size = cv2.bitwise_and(img_copy, img_copy, mask=rectangle_mask_gray)
    return gt_mask, out_origin_size, img_copy[y11:y22, x11:x22]


def mask_object_count_and_ratio(gt_path):
    gt_mask = cv2.imread(gt_path, 0)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    ratio_object = (1.0 - float(stats[0][4]) / float(gt_mask.size))
    return retval, labels, stats, centroids, ratio_object


def rebuild_masked_dataset(dataset_dic, model):
    data_set = GetDataset(dataset_dic.get("image_root"), dataset_dic.get("depth_root"),
                          dataset_dic.get("gt_root"))
    pbar = tqdm(range(len(data_set)), desc="rebuild_dataset:", unit='img')
    date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    save_dir = "../testResults/" + date_str
    dataset_writer = SummaryWriter(save_dir + "/log", comment="log")
    f = open("../myDocument/imageNet1k-class.txt")
    class_list = f.readlines()
    gt_dir = osp.join(save_dir, 'data', 'gt')
    mask_origin_dir = osp.join(save_dir, 'data', 'mask_origin')
    mask_resize_dir = osp.join(save_dir, 'data', 'mask_resize')
    merge_img_dir = osp.join(save_dir, 'data', 'merge_img')
    if not osp.isdir(gt_dir):
        os.makedirs(gt_dir)
    if not osp.isdir(mask_origin_dir):
        os.makedirs(mask_origin_dir)
    if not osp.isdir(mask_resize_dir):
        os.makedirs(mask_resize_dir)
    if not osp.isdir(merge_img_dir):
        os.makedirs(merge_img_dir)
    for idx in pbar:
        name, image_path, depth_path, gt_path, img_array = data_set.load_data(idx)
        # 根据mask切割
        gt_mask, out_origin_size, img_copy_resize = draw_max_rec_of_mask(gt_path, image_path)
        # 统计联通区域和显著性物体占比
        retval, labels, stats, centroids, ratio_object = mask_object_count_and_ratio(gt_path)
        img_resize_salient_object_path = osp.join(mask_resize_dir, name)
        img_mask_origin_path = osp.join(mask_origin_dir, name)
        cv2.imwrite(osp.join(gt_dir, name[:-4] + '.png'), gt_mask)
        cv2.imwrite(img_mask_origin_path, out_origin_size)
        cv2.imwrite(img_resize_salient_object_path, img_copy_resize)

        # 原图像的结果
        result_origin = inference_model(model, img_array)
        # 显著性目标最大外切正方形的测试结果
        result_resize = inference_model(model, img_copy_resize)
        for class_info in class_list:
            class_info_list = class_info.split(" ")
            if str(result_origin.get("pred_label")) == class_info_list[0]:
                result_origin["pred_hypernyms_class"] = class_info_list[2].strip(",")
            if str(result_resize.get("pred_label")) == class_info_list[0]:
                result_resize["pred_hypernyms_class"] = class_info_list[2].strip(",")
            if result_origin.get("pred_label") == result_resize.get("pred_label"):
                same_flag = 'T'
            else:
                same_flag = 'F'
        hparam_dic = {
            "file_pth": image_path,
            "pred_class": result_origin.get("pred_class"),
            "pred_hypernyms_class": result_origin.get("pred_hypernyms_class"),
            "pred_class_salient_object": result_resize.get("pred_class"),
            "pred_hypernyms_class_salient_object": result_resize.get("pred_hypernyms_class"),
            "same_flag": same_flag
        }
        metric_dic = {
            "pred_label": result_origin.get("pred_label"),
            "pred_score": result_origin.get("pred_score"),
            "pred_label_salient_object": result_resize.get("pred_label"),
            "pred_score_salient_object": result_resize.get("pred_score"),
            "object_count": retval - 1,
            "object_ratio": ratio_object
        }
        dataset_writer.add_hparams(hparam_dic, metric_dic, name='log/' + image_path)
        merge_img_list = [image_path,
                          depth_path,
                          gt_path,
                          img_mask_origin_path]
        tag_dic = {
            'class_name': result_origin.get("pred_class"),
            'pred_socre': result_origin.get("pred_score"),
            'ob_class_name': result_resize.get("pred_class"),
            'ob_pred_socre': result_resize.get("pred_score"),
            'object_ratio': ratio_object,
            'object_count': retval - 1,
            'same_flag': same_flag,
            'name': name[:-4]
        }
        image_grid = utils.merge_image_file_name_return_tensor(merge_img_list, name[:-4] + '_merge.png', merge_img_dir,
                                                               len(merge_img_list), tag_dic)
        dataset_writer.add_image(dataset_dic["dataset_name"], image_grid, global_step=idx)


if __name__ == '__main__':
    # conformer
    config_file = '../configs/conformer/conformer-tiny-p16_8xb128_in1k_mine.py'
    # 权重文件参数路径
    checkpoint_file = '../checkpoints/conformer-tiny-p16_3rdparty_8xb128_in1k_20211206-f6860372.pth'
    # 设置设备为GPU 或者 device='cpu'

    device = 'cuda:0'
    dataset_dic_train = {
        'image_root': '../data/COME15K/train/imgs_right/',
        'gt_root': '../data/COME15K/train/gt_right/',
        'depth_root': '../data/COME15K/train/depths/',
        'dataset_name': 'come15k_train'
    }

    dataset_dic_val_e = {
        'image_root': '../data/COME15K/val/' + 'COME-E' + '/RGB/',
        'gt_root': '../data/COME15K/val/' + 'COME-E' + '/GT/',
        'depth_root': '../data/COME15K/val/' + 'COME-E' + '/depths/',
        'dataset_name': 'come15k_val_easy'
    }
    dataset_dic_val_h = {
        'image_root': '../data/COME15K/val/' + 'COME-H' + '/RGB/',
        'gt_root': '../data/COME15K/val/' + 'COME-H' + '/GT/',
        'depth_root': '../data/COME15K/val/' + 'COME-H' + '/depths/',
        'dataset_name': 'come15k_val_hard'
    }
    dataset_dic_test_e = {
        'image_root': '../data/COME15K/test/' + 'COME-E' + '/RGB/',
        'gt_root': '../data/COME15K/test/' + 'COME-E' + '/GT/',
        'depth_root': '../data/COME15K/test/' + 'COME-E' + '/depths/',
        'dataset_name': 'come15k_test_easy'
    }
    dataset_dic_test_h = {
        'image_root': '../data/COME15K/test/' + 'COME-H' + '/RGB/',
        'gt_root': '../data/COME15K/test/' + 'COME-H' + '/GT/',
        'depth_root': '../data/COME15K/test/' + 'COME-H' + '/depths/',
        'dataset_name': 'come15k_test_hard'
    }
    model = init_my_model(config_file, checkpoint_file, device)
    # 标注图像类型
    # rebuild_dataset(dataset_dic_train, model)
    # 切割图像类型并标注类型和显著图像占比
    rebuild_masked_dataset(dataset_dic_train, model)
