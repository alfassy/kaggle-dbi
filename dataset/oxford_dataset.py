from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

breeds = {
    'abyssinian': [],
    'american_bulldog': [],
    'american_pit_bull_terrier': [],
    'basset_hound': [],
    'beagle': [],
    'bengal': [],
    'birman': [],
    'bombay': [],
    'boxer': [],
    'british_shorthair': [],
    'chihuahua': [],
    'egyptian_mau': [],
    'english_cocker_spaniel': [],
    'english_setter': [],
    'german_shorthaired': [],
    'great_pyrenees': [],
    'havanese': [],
    'japanese_chin': [],
    'keeshond': [],
    'leonberger': [],
    'maine_coon': [],
    'miniature_pinscher': [],
    'newfoundland': [],
    'persian': [],
    'pomeranian': [],
    'pug': [],
    'ragdoll': [],
    'russian_blue': [],
    'saint_bernard': [],
    'samoyed': [],
    'scottish_terrier': [],
    'shiba_inu': [],
    'siamese': [],
    'sphynx': [],
    'staffordshire_bull_terrier': [],
    'wheaten_terrier': [],
    'yorkshire_terrier': []
}

category_to_int = {
    'abyssinian': 1,
    'american_bulldog': 2,
    'american_pit_bull_terrier': 3,
    'basset_hound': 4,
    'beagle': 5,
    'bengal': 6,
    'birman': 7,
    'bombay': 8,
    'boxer': 9,
    'british_shorthair': 10,
    'chihuahua': 11,
    'egyptian_mau': 12,
    'english_cocker_spaniel': 13,
    'english_setter': 14,
    'german_shorthaired': 15,
    'great_pyrenees': 16,
    'havanese': 17,
    'japanese_chin': 18,
    'keeshond': 19,
    'leonberger': 20,
    'maine_coon': 21,
    'miniature_pinscher': 22,
    'newfoundland': 23,
    'persian': 24,
    'pomeranian': 25,
    'pug': 26,
    'ragdoll': 27,
    'russian_blue': 28,
    'saint_bernard': 29,
    'samoyed': 30,
    'scottish_terrier': 31,
    'shiba_inu': 32,
    'siamese': 33,
    'sphynx': 34,
    'staffordshire_bull_terrier': 35,
    'wheaten_terrier': 36,
    'yorkshire_terrier': 37
}

int_to_category = {
    1: 'abyssinian',
    2: 'american_bulldog',
    3: 'american_pit_bull_terrier',
    4:  'basset_hound',
    5:  'beagle',
    6:  'bengal',
    7:  'birman',
    8:  'bombay',
    9:  'boxer',
    10: 'british_shorthair',
    11: 'chihuahua',
    12: 'egyptian_mau',
    13: 'english_cocker_spaniel',
    14: 'english_setter',
    15: 'german_shorthaired',
    16: 'great_pyrenees',
    17: 'havanese',
    18: 'japanese_chin',
    19: 'keeshond',
    20: 'leonberger',
    21: 'maine_coon',
    22: 'miniature_pinscher',
    23: 'newfoundland',
    24: 'persian',
    25: 'pomeranian',
    26: 'pug',
    27: 'ragdoll',
    28: 'russian_blue',
    29: 'saint_bernard',
    30: 'samoyed',
    31: 'scottish_terrier',
    32: 'shiba_inu',
    33: 'siamese',
    34: 'sphynx',
    35: 'staffordshire_bull_terrier',
    36: 'wheaten_terrier',
    37: 'yorkshire_terrier'
}

dog_breeds_int_2_category = {
    1: 'american_bulldog',
    2: 'american_pit_bull_terrier',
    3: 'basset_hound',
    4: 'beagle',
    5: 'boxer',
    6: 'chihuahua',
    7: 'english_cocker_spaniel',
    8: 'english_setter',
    9: 'german_shorthaired',
    10: 'great_pyrenees',
    11: 'havanese',
    12: 'japanese_chin',
    13: 'keeshond',
    14: 'leonberger',
    15: 'miniature_pinscher',
    16: 'newfoundland',
    17: 'pomeranian',
    18: 'pug',
    19: 'saint_bernard',
    20: 'samoyed',
    21: 'scottish_terrier',
    22: 'shiba_inu',
    23: 'staffordshire_bull_terrier',
    24: 'wheaten_terrier',
    25: 'yorkshire_terrier'
}

dog_breeds_category_to_int = {
    'american_bulldog': 1,
    'american_pit_bull_terrier': 2,
    'basset_hound': 3,
    'beagle': 4,
    'boxer': 5,
    'chihuahua': 6,
    'english_cocker_spaniel': 7,
    'english_setter': 8,
    'german_shorthaired': 9,
    'great_pyrenees': 10,
    'havanese': 11,
    'japanese_chin': 12,
    'keeshond': 13,
    'leonberger': 14,
    'miniature_pinscher': 15,
    'newfoundland': 16,
    'pomeranian': 17,
    'pug': 18,
    'saint_bernard': 19,
    'samoyed': 20,
    'scottish_terrier': 21,
    'shiba_inu': 22,
    'staffordshire_bull_terrier': 23,
    'wheaten_terrier': 24,
    'yorkshire_terrier': 25
}


class oxford_dataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.root_folder = img_folder
        self.all_img_paths = [img_path for img_path in self.root_folder.iterdir()]
        self.transform = transform
        self.class_2_idx = dog_breeds_category_to_int
        self.idx_2_class = dog_breeds_int_2_category
        # check that all the imgs has labels in the csv
        labels_names_list = []
        labels_int_list = []
        dogs_img_paths_list = []
        for path in self.all_img_paths:
            label_name = str(((path.name.split('.')[0]).rsplit('_', 1))[0])
            # don't use cats images
            if label_name not in self.class_2_idx.keys():
                continue
            # don't use greyscale images (only 2 images out of 4990)
            data = Image.open(path)
            if data.mode != 'RGB':
                continue
            dogs_img_paths_list.append(path)
            labels_names_list.append(label_name)
            label_int = self.class_2_idx[label_name]
            labels_int_list.append(label_int)
        self.labels_names = labels_names_list
        self.labels_int_list = labels_int_list
        self.dogs_img_paths_list = dogs_img_paths_list
        self.img_num = len(self.dogs_img_paths_list)
        # randInd = np.random.randint(0, self.img_num, size=10)
        # print([self.labels_names[i] for i in randInd])
        # print([self.labels_int_list[i] for i in randInd])
        # print([self.dogs_img_paths_list[i] for i in randInd])
        print('dataset size: ', self.img_num)

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        img_path = self.dogs_img_paths_list[index]
        label = self.labels_int_list[index]
        data = Image.open(img_path)
        if self.transform:
            data = self.transform(data)
        return data, label
