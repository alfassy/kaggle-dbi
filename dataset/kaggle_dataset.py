from torch.utils.data import Dataset
from PIL import Image


'''
kaggle dog breed identification dataset
'''


class DBI_dataset(Dataset):
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
        print(self.img_num)
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