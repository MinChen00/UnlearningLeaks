import os
from os import path
import pickle
import logging
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, MNIST, CIFAR100, STL10, ImageFolder
import torchvision.transforms as transforms

import config


class LoadData:
    def __init__(self):
        self.logger = logging.getLogger("load_data")

    @staticmethod
    def load_location(original_label):
        if original_label == "NY":
            df = pickle.load(open(config.PROCESSED_DATASET_PATH + "Insta_ny", 'rb'))
        elif original_label == "LA":
            df = pickle.load(open(config.PROCESSED_DATASET_PATH + "Insta_la", 'rb'))
        else:
            raise Exception("invalid location city name")
        return df

    @staticmethod
    def load_adult(original_label):
        df = pickle.load(open(config.PROCESSED_DATASET_PATH + "adult", 'rb'))
        if original_label == 'income':
            df = df[['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                     'occupation', 'relationship', 'marital-status', 'race', 'gender', 'capital-gain',
                     'capital-loss', 'hours-per-week', 'native-country', 'income']]
        return df

    @staticmethod
    def load_accident(original_label):
        df = pickle.load(open(config.PROCESSED_DATASET_PATH + "accident", 'rb'))
        # 3-class balanced
        if original_label == 'severity':
            df = df[['Source', 'TMC', 'Start_Lat', 'Start_Lng', 'Distance(mi)',
                     'Side', 'County', 'State', 'Timezone', 'Airport_Code', 'Temperature(F)',
                     'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                     'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)',
                     'Weather_Condition', 'Amenity', 'Crossing', 'Junction', 'Railway',
                     'Station', 'Traffic_Signal', 'Sunrise_Sunset', 'Civil_Twilight',
                     'Nautical_Twilight', 'Astronomical_Twilight', 'Severity']]
            df['Severity'] = df['Severity'].replace(2, 1)
            df['Severity'] = df['Severity'].replace(4, 2)
            df['Severity'] = df['Severity'].replace(3, 2)
        return df

    def load_mnist_data(self):
        trainloader, testloader, trainset, testset = LoadData.load_mnist()
        return trainset

    def load_cifar10_data(self):
        train_loader, test_loader, train_set, test_set = LoadData.load_cifar10()
        return train_set

    def load_stl10_data(self):
        train_loader, test_loader, train_set, test_set = LoadData.load_stl10()
        return train_set

    @staticmethod
    def loader_cat_data(dataset, original_label, batch_size):
        if dataset == 'adult':
            df = LoadData.load_adult("income")
        elif dataset == 'accident':
            df = LoadData.load_accident(original_label='severity')
            train_size = df.shape[0]
            data = df.iloc[:, :-1].to_numpy()
            labels = df.iloc[:, -1].to_numpy()
            zero_indices = np.where(labels == 2)
            labels[zero_indices] = 0
            train_x = torch.tensor(torch.from_numpy(np.array(data[:train_size, :], dtype=np.float32)))
            train_y = torch.tensor(np.int64(labels[:train_size]))
            train_dset = TensorDataset(train_x, train_y)
            return train_dset
        elif dataset == 'location':
            df = LoadData.load_location(original_label)
        else:
            raise Exception("invalid dataset name")

        train_size = df.shape[0]
        data = df.iloc[:, :-1].to_numpy()
        labels = df.iloc[:, -1].to_numpy()
        train_x = torch.tensor(data[:train_size, :]).float()
        train_y = torch.tensor(np.int64(labels[:train_size]))
        train_dset = TensorDataset(train_x, train_y)

        return train_dset

    @staticmethod
    def load_mnist(batch_size=32):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = MNIST(root=config.ORIGINAL_DATASET_PATH + 'mnist', train=True, transform=transform,
                          download=True)
        test_set = MNIST(root=config.ORIGINAL_DATASET_PATH + 'mnist', train=False, transform=transform)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def load_cifar10(batch_size=32, num_workers=1):
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = CIFAR10(root=config.ORIGINAL_DATASET_PATH + 'cifar10', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_set = CIFAR10(root=config.ORIGINAL_DATASET_PATH + 'cifar10', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def load_stl10(batch_size=32, num_workers=1):
        train_set = STL10(root=config.ORIGINAL_DATASET_PATH + 'stl10', split='train', download=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.Resize(32),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_set = STL10(root=config.ORIGINAL_DATASET_PATH + 'stl10', split='test', download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize(32),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def load_image(dataset_name):
        load_data = LoadData()
        if dataset_name == 'mnist':
            return load_data.load_mnist()
        elif dataset_name == 'stl10':
            return load_data.load_stl10()
        elif dataset_name == 'cifar10':
            return load_data.load_cifar10()


class DataStore:
    def __init__(self, args):
        self.logger = logging.getLogger("DataStore")
        self.args = args
        self.determine_data_path()

    def create_basic_folders(self):
        folder_list = [config.SPLIT_INDICES_PATH, config.SHADOW_MODEL_PATH, config.TARGET_MODEL_PATH,
                       config.ATTACK_DATA_PATH, config.ATTACK_MODEL_PATH]
        for folder in folder_list:
            self.create_folder(folder)

    def determine_data_path(self):
        self.save_name = "_".join((self.args['unlearning_method'], self.args['dataset_name'],
                                   self.args['original_label'], self.args['original_model'],
                                   str(self.args['shadow_set_num']),
                                   str(self.args['target_set_num']),
                                   str(self.args['shadow_set_size']),
                                   str(self.args['target_set_size']),
                                   str(self.args['shadow_unlearning_size']),
                                   str(self.args['target_unlearning_size']),
                                   str(self.args['shadow_unlearning_num']),
                                   str(self.args['target_unlearning_num']),
                                   str(self.args['target_num_shard']),
                                   str(self.args['shadow_num_shard'])
                                   ))
        if self.args['is_dp_defense']:
            self.save_name += "_DP"

        self.target_model_name = config.TARGET_MODEL_PATH + self.save_name
        self.shadow_model_name = config.SHADOW_MODEL_PATH + self.save_name

        self.attack_train_data = config.SHADOW_MODEL_PATH + "posterior" + self.save_name
        self.attack_test_data = config.TARGET_MODEL_PATH + "posterior" + self.save_name

    def load_raw_data(self):
        load = LoadData()
        num_classes = {
            "adult": 2,
            "accident": 3,
            "location": 9,
            "cifar10": 10,
            "mnist": 10,
            "stl10": 10
        }
        self.num_classes = num_classes[self.args['dataset_name']]
        if self.args['dataset_name'] == "cifar10":
            self.df = load.load_cifar10_data()
            self.num_records = self.df.data.shape[0]
        elif self.args['dataset_name'] == "stl10":
            self.df = load.load_stl10_data()
            self.num_records = self.df.data.shape[0]
        elif self.args['dataset_name'] == "mnist":
            self.df = load.load_mnist_data()
            self.num_records = self.df.data.shape[0]
        # Uncomment this to test categorical dataset on DNN model    
        # elif self.args['dataset_name'] in ["adult", "accident", "location"]:
        #     self.df = load.loader_cat_data(self.args['dataset_name'], self.args['original_label'], batch_size=32)
        #     self.num_records = self.df.tensors[0].data.shape[0]
        elif self.args['dataset_name'] == "adult":
            self.df = load.load_adult(self.args['original_label'])
            self.num_records = self.df.shape[0]        
        elif self.args['dataset_name'] == "accident":
            self.df = load.load_accident(self.args['original_label'])
            self.num_records = self.df.shape[0]    
        elif self.args['dataset_name'] == "location":
            self.df = load.load_location(self.args['original_label'])
            self.num_records = self.df.shape[0]        
        else:
            raise Exception("invalid dataset name")

        return self.df, self.num_records, self.num_classes

    def save_raw_data(self):
        pass

    def save_record_split(self, record_split):
        pickle.dump(record_split, open(config.SPLIT_INDICES_PATH + self.save_name, 'wb'))

    def load_record_split(self):
        record_split = pickle.load(open(config.SPLIT_INDICES_PATH + self.save_name, 'rb'))
        return record_split

    def save_attack_train_data(self, attack_train_data):
        pickle.dump((attack_train_data), open(self.attack_train_data, 'wb'))

    def load_attack_train_data(self):
        attack_train_data = pickle.load(open(self.attack_train_data, 'rb'))
        return attack_train_data

    def save_attack_test_data(self, attack_test_data):
        pickle.dump((attack_test_data), open(self.attack_test_data, 'wb'))

    def load_attack_test_data(self):
        attack_test_data = pickle.load(open(self.attack_test_data, 'rb'))
        return attack_test_data

    def create_folder(self, folder):
        if not path.exists(folder):
            try:
                self.logger.info("checking directory %s", folder)
                os.mkdir(folder)
                self.logger.info("new directory %s created", folder)
            except OSError as error:
                self.logger.info("deleting old and creating new empty %s", folder)
                # os.rmdir(folder)
                shutil.rmtree(folder)
                os.mkdir(folder)
                self.logger.info("new empty directory %s created", folder)
        else:
            self.logger.info("folder %s exists, do not need to create again.", folder)
