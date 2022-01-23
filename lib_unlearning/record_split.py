import numpy as np
from sklearn.model_selection import train_test_split


class RecordSplit:
    def __init__(self, num_records, args):
        self.args = args
        self.num_records = num_records
        self.shadow_set = {}
        self.target_set = {}

    def split_shadow_target(self):
        shadow_indices, target_indices = train_test_split(np.arange(self.num_records), test_size=0.5, random_state=100)
        self.shadow_positive_indices, self.shadow_negative_indices = \
            train_test_split(shadow_indices, test_size=0.2, random_state=100)
        self.target_positive_indices, self.target_negative_indices = \
            train_test_split(target_indices, test_size=0.2, random_state=100)
        self.target_negative_indices, self.target_replace_indices = \
            train_test_split(self.target_negative_indices, test_size=0.15, random_state=0)
        self.shadow_negative_indices, self.shadow_replace_indices = \
            train_test_split(self.shadow_negative_indices, test_size=0.15, random_state=0)

    def sample_records(self, unlearning_method):
        if unlearning_method == "scratch":
            self._sample_records_scratch()
        elif unlearning_method == "sisa":
            self._sample_records_sisa()
        else:
            raise Exception("invalid unlearning method")

    def _sample_records_scratch(self):
        for i in range(self.args['shadow_set_num']):
            shadow_set_indices = np.random.choice(self.shadow_positive_indices, self.args['shadow_set_size'], replace=False)
            shadow_unlearning_set = {}

            for index, unlearning_num in enumerate(range(self.args['shadow_unlearning_size'])):
                shadow_unlearning_indices = np.random.choice(shadow_set_indices, self.args['shadow_unlearning_num'], replace=False)
                shadow_unlearning_set[index] = shadow_unlearning_indices

            self.shadow_set[i] = {
                "set_indices": shadow_set_indices,
                "unlearning_set": shadow_unlearning_set
            }

        for i in range(self.args['target_set_num']):
            target_set_indices = np.random.choice(self.target_positive_indices, self.args['target_set_size'], replace=False)
            target_unlearning_set = {}

            for index, unlearning_num in enumerate(range(self.args['target_unlearning_size'])):
                target_unlearning_indices = np.random.choice(target_set_indices, self.args['target_unlearning_num'], replace=False)
                target_unlearning_set[index] = target_unlearning_indices

            self.target_set[i] = {
                "set_indices": target_set_indices,
                "unlearning_set": target_unlearning_set
            }

    def _sample_records_sisa(self):
        for i in range(self.args['shadow_set_num']):
            shadow_set_indices = np.random.choice(self.shadow_positive_indices, self.args['shadow_set_size'], replace=False)
            shadow_unlearning_indices = np.random.choice(shadow_set_indices, self.args['shadow_unlearning_size'], replace=False)
            shard_set = np.reshape(shadow_set_indices, (self.args['shadow_num_shard'], -1))
            unlearning_shard_mapping = {}

            for index in shadow_unlearning_indices:
                unlearning_shard_mapping[index] = np.where(shard_set == index)[0][0]

            self.shadow_set[i] = {
                "set_indices": shadow_set_indices,
                "shard_set": shard_set,
                "unlearning_indices": shadow_unlearning_indices,
                "unlearning_shard_mapping": unlearning_shard_mapping
            }

        for i in range(self.args['target_set_num']):
            target_set_indices = np.random.choice(self.target_positive_indices, self.args['target_set_size'], replace=False)
            target_unlearning_indices = np.random.choice(target_set_indices, self.args['target_unlearning_size'], replace=False)
            shard_set = np.reshape(target_set_indices, (self.args['target_num_shard'], -1))
            unlearning_shard_mapping = {}

            for index in target_unlearning_indices:
                unlearning_shard_mapping[index] = np.where(shard_set == index)[0][0]

            self.target_set[i] = {
                "set_indices": target_set_indices,
                "shard_set": shard_set,
                "unlearning_indices": target_unlearning_indices,
                "unlearning_shard_mapping": unlearning_shard_mapping
            }

    def generate_sample(self, sample_name="shadow"):
        if sample_name == "shadow":
            self.negative_indices = self.shadow_negative_indices
            self.replace_indices = self.shadow_replace_indices
            self.sample_set = self.shadow_set
        elif sample_name == "target":
            self.negative_indices = self.target_negative_indices
            self.replace_indices = self.target_replace_indices
            self.sample_set = self.target_set
        else:
            raise Exception("invalid sample name")
