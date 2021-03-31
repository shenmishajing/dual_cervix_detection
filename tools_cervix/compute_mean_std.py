from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import glob
import cv2
import os 
from tqdm import tqdm
import numpy as np

def load_case_id_from_txt(case_id_path):
    with open(case_id_path, "r") as f:
        case_id_set = set([
            line.strip() for line in f.readlines() if len(line.strip()) > 0
        ])
    return case_id_set


class ImgDataset(Dataset):
    def __init__(self, img_path_list, img_size=(512, 512)):
        self.img_size = img_size
        self.paths = img_path_list


    def __getitem__(self, index):
        path = self.paths[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
        img = cv2.resize(img, self.img_size)
        img = img.transpose(2, 0, 1) #通道数放在第一维

        return img

    def __len__(self):
        return len(self.paths)


def stat_mean_std(case_id_path, img_dir):
    case_id_list = load_case_id_from_txt(case_id_path)
    acid_path_list = [os.path.join(img_dir, case_id + "_2.jpg") for case_id in case_id_list]
    iodine_path_list = [os.path.join(img_dir, case_id + "_3.jpg") for case_id in case_id_list]

    acid_dataset = ImgDataset(acid_path_list)
    dataloader = DataLoader(acid_dataset, batch_size=32,
                            num_workers=32, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in tqdm(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    mean /= 255.0
    std /= 255.0
    print('Acid mean', mean)
    print('Acid std', std)

    iodine_dataset = ImgDataset(iodine_path_list)
    dataloader = DataLoader(iodine_dataset, batch_size=32,
                            num_workers=32, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in tqdm(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    mean /= 255.0
    std /= 255.0

    print("Iodine mean", mean)
    print("Iodine std", std)


if __name__ == '__main__':
    # total_case_id_path = "/data/luochunhua/cervix/cervix_det_data/data_split/total.txt"
    train_case_id_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/train.txt"
    # valid_case_id_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/valid.txt"
    img_dir = "/data/luochunhua/cervix/cervix_det_data/img"
    print("train")
    stat_mean_std(train_case_id_path, img_dir)
