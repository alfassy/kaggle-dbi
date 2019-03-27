from torch.utils.data import Dataset
from PIL import Image
from dataset.oxford_dataset import oxford_dataset
from pathlib import Path


'''
kaggle dog breed identification dataset
'''


class DBI_dataset(Dataset):
    def __init__(self, img_folder, df_train, df_test, use_oxford, is_train=True, transform=None):
        self.root_folder = img_folder
        self.labels_list = list(df_test.columns)
        self.imgs = [img_path for img_path in self.root_folder.iterdir()]
        self.transform = transform
        self.weak_classes_from_oxford = ['beagle', 'staffordshire_bullterrier', 'yorkshire_terrier']
        if use_oxford == 1:
            self.use_oxford = True
        else:
            self.use_oxford = False
        # check that all the imgs has labels in the csv
        if is_train:
            img_paths_list = []
            for img_path in self.imgs:
                if img_path.name.split('.')[0] in list(df_train.index):
                    img_paths_list.append(img_path)
            self.imgs_paths = img_paths_list
        else:
            self.imgs_paths = self.imgs
        self.img_num = len(self.imgs_paths)
        print("kaggle dataset size without oxford: ", self.img_num)
        self.class_2_idx = {self.labels_list[i]: i for i in range(len(self.labels_list))}
        self.idx_2_class = {v: k for k, v in self.class_2_idx.items()}
        self.is_train = is_train
        self.test_names = df_test.index
        if self.is_train:
            self.train_targets = [self.class_2_idx[df_train.loc[self.imgs_paths[i].name.split('.')[0]]['breed']]
                                  for i in range(self.img_num)]
            if self.use_oxford:
                oxford_imgs_path = Path('./data/oxford_pets/images')
                oxfordDataset = oxford_dataset(oxford_imgs_path, transform=self.transform)
                for i in range(oxfordDataset.img_num):
                    label_name = oxfordDataset.oxford_dog_breeds_to_kaggle[oxfordDataset.labels_names[i]]
                    if label_name in self.weak_classes_from_oxford:
                        self.imgs_paths.append(oxfordDataset.dogs_img_paths_list[i])
                        self.train_targets.append(self.class_2_idx[label_name])
        self.img_num_with_oxford = len(self.imgs_paths)
        print("kaggle dataset size with oxford: ", self.img_num_with_oxford)

    def __len__(self):
        return self.img_num_with_oxford

    def __getitem__(self, index):
        img_path = self.imgs_paths[index]
        if self.is_train:
            label = self.train_targets[index]
        else:
            label = self.test_names[index]
        data = Image.open(img_path)
        if self.transform:
            data = self.transform(data)
        return data, label


class DBI_dataset_ensemble(Dataset):
    def __init__(self, img_folder, df_train, df_test, is_train=True, transform=None):
        self.root_folder = img_folder
        self.labels_list = list(df_test.columns)

        self.imgs = [img_path for img_path in self.root_folder.iterdir()]
        self.transform = transform
        # check that all the imgs has labels in the csv
        if is_train:
            img_list = []
            for img_path in self.imgs:
                if img_path.name.split('.')[0] in list(df_train.index):
                    img_list.append(img_path)
            self.imgs = img_list
        self.img_num = len(self.imgs)
        self.class_2_idx = {self.labels_list[i]: i for i in range(len(self.labels_list))}
        self.idx_2_class = {v: k for k, v in self.class_2_idx.items()}
        self.is_train = is_train
        self.test_names = df_test.index

        if self.is_train:
            self.train_targets = [self.class_2_idx[df_train.loc[self.imgs[i].name.split('.')[0]]['breed']]
                                  for i in range(self.img_num)]

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.is_train:
            label = self.train_targets[index]
        else:
            label = self.test_names[index]
        data = Image.open(img_path)
        if self.transform:
            data = self.transform(data)
        return data, label