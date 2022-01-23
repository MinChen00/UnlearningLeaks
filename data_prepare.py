import logging
import os
import json
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import config


class PreprocessAccident:
    def __init__(self):
        self.logger = logging.getLogger("preprocess_accident")
        self.attrs = []
        self.shape = []

    def load_data(self):
        self.logger.info("loading dataset")
        self.df = pd.read_csv(config.ORIGINAL_DATASET_PATH + "accident.csv", low_memory=False, nrows=600000)
        self.specs = json.load(open(config.ORIGINAL_DATASET_PATH + "accident-specs.json", 'r'))

    def transform_to_num(self):
        for col in self.df.columns:
            if self.specs[col]["include"]:
                self.logger.info("transforming %s" % (col,))
                data = np.zeros(self.df.shape[0], dtype=np.uint32)

                if self.specs[col]["type"] == "bool":
                    data = self.df[col]
                    self.shape.append(2)

                elif self.specs[col]["type"] == "enum":
                    unique = pd.unique(self.df[col])
                    for index, value in enumerate(unique):
                        data[np.where(self.df[col] == value)[0]] = index + 1
                    self.shape.append(unique.size + 1)

                elif self.specs[col]["type"] == "float":
                    min_value = np.min(self.df[col])
                    max_value = np.max(self.df[col])
                    bin_value = np.linspace(min_value, max_value, 100 + 1)
                    bin_value[-1] += 1
                    data = np.digitize(self.df[col], bin_value)
                    self.shape.append(102)

                elif self.specs[col]["type"] == "int":
                    max_value = np.max(self.df[col])
                    for value in range(int(max_value) + 1):
                        data[np.where(self.df[col] == value)[0]] = value

                    self.shape.append(int(max_value) + 1)

                else:
                    raise Exception("invalid data type")

                self.df[col] = data
                self.attrs.append(col)

    def generate_specs(self, df):
        specs_dict = defaultdict(dict)

        for col in df.columns:
            if df.dtypes[col] == "float64":
                specs_dict[col]["type"] = "float"
            elif df.dtypes[col] == "int64":
                specs_dict[col]["type"] = "int"
            elif df.dtypes[col] == "object":
                specs_dict[col]["type"] = "enum"
            elif df.dtypes[col] == "bool":
                specs_dict[col]["type"] = "bool"
            else:
                self.logger.info("wrong type")

            specs_dict[col]["include"] = True

        json.dump(specs_dict, open(config.ORIGINAL_DATASET_PATH + "accident-specs.json", 'w'), indent=4)

    def save_data(self):
        self.logger.info("saving data")
        df = self.df[self.attrs].astype(np.uint32)
        pickle.dump(df, open(config.PROCESSED_DATASET_PATH + "accident", 'wb'))


class PreprocessAdult:
    def __init__(self):
        pass

    def process_adult(self):
        adult = pd.read_csv(config.ORIGINAL_DATASET_PATH + 'adult.csv')

        # fill null variable
        var = adult['native-country'].mode()
        adult['native-country'] = adult['native-country'].replace(np.NaN, var[0])

        var1 = adult.workclass.mode()[0]
        adult.workclass = adult.workclass.replace(np.NaN, var1)

        var2 = adult.occupation.mode()[0]
        adult.occupation = adult.occupation.replace(np.NaN, var2)

        # convert string into integer
        le = LabelEncoder()
        cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'native-country', 'income']
        for col in cols:
            adult[col] = le.fit_transform(adult[col])
        pickle.dump(adult, open(config.PROCESSED_DATASET_PATH + "adult", 'wb'))

    def process_location(self):
        Insta_ny = pd.read_csv(config.ORIGINAL_DATASET_PATH + 'location/ny_withCatId.csv')
        Insta_la = pd.read_csv(config.ORIGINAL_DATASET_PATH + 'location/la_withCatId.csv')

        Insta_ny = Insta_ny.drop(columns="locid")
        Insta_la = Insta_la.drop(columns="locid")
        pickle.dump(Insta_ny, open(config.PROCESSED_DATASET_PATH + "Insta_ny", 'wb'))
        pickle.dump(Insta_la, open(config.PROCESSED_DATASET_PATH + "Insta_la", 'wb'))


if __name__ == "__main__":
    os.chdir("../")

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    preprocess = PreprocessAccident()
    preprocess.load_data()
    preprocess.transform_to_num()
    preprocess.save_data()

    preprocess = PreprocessAdult()
    preprocess.process_adult()
    preprocess.process_location()
