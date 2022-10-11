import datetime
from nltk.corpus import wordnet2021 as wn
from tensorboardX import SummaryWriter
from myTools.data import GetDataset
from mmcls.apis import init_model, inference_model, show_result_pyplot
from tqdm import tqdm




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

    for idx in pbar:
        name, image_path, depth_path, gt_path, img_array = data_set.load_data(idx)
        result = inference_model(model, img_array)
        hparam_dic = {
            "file_pth": image_path,
            "pred_class":  result.get("pred_class")
        }
        metric_dic = {
            "pred_label": result.get("pred_label"),
            "pred_score": result.get("pred_score")
        }
        dataset_writer.add_hparams(hparam_dic, metric_dic, name=image_path)
        # dataset_writer.add_image()
        # show_result_pyplot(model, image_path, result)


if __name__ == '__main__':
    # conformer
    # config_file = '../configs/conformer/conformer-tiny-p16_8xb128_in1k_mine.py'
    # # 权重文件参数路径
    # checkpoint_file = '../checkpoints/conformer-tiny-p16_3rdparty_8xb128_in1k_20211206-f6860372.pth'
    # conformer
    # config_file = '../configs/conformer/conformer-tiny-p16_8xb128_in1k_mine.py'
    # # 权重文件参数路径
    # checkpoint_file = '../checkpoints/conformer-tiny-p16_3rdparty_8xb128_in1k_20211206-f6860372.pth'
    # 设置设备为GPU 或者 device='cpu'
    device = 'cuda:0'
    dataset_dic_train = {
        'image_root': '../data/COME15K/train/imgs_right/',
        'gt_root': '../data/COME15K/train/gt_right/',
        'depth_root': '../data/COME15K/train/depths/'
    }

    dataset_dic_val_e = {
        'image_root': '../data/COME15K/val/' + 'COME-E' + '/RGB/',
        'gt_root': '../data/COME15K/val/' + 'COME-E' + '/GT/',
        'depth_root': '../data/COME15K/val/' + 'COME-E' + '/depths/',

    }
    dataset_dic_val_h = {
        'image_root': '../data/COME15K/val/' + 'COME-H' + '/RGB/',
        'gt_root': '../data/COME15K/val/' + 'COME-H' + '/GT/',
        'depth_root': '../data/COME15K/val/' + 'COME-H' + '/depths/'
    }
    dataset_dic_test_e = {
        'image_root': '../data/COME15K/test/' + 'COME-E' + '/RGB/',
        'gt_root': '../data/COME15K/test/' + 'COME-E' + '/GT/',
        'depth_root': '../data/COME15K/test/' + 'COME-E' + '/depths/'
    }
    dataset_dic_test_h = {
        'image_root': '../data/COME15K/test/' + 'COME-H' + '/RGB/',
        'gt_root': '../data/COME15K/test/' + 'COME-H' + '/GT/',
        'depth_root': '../data/COME15K/test/' + 'COME-H' + '/depths/'
    }
    model = init_my_model(config_file, checkpoint_file, device)
    rebuild_dataset(dataset_dic_train, model)

