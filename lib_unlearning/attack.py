import logging
import numpy as np

from models import MLP, DT, RF, LR


class Attack:
    def __init__(self, attack_model_name, shadow_post_df, target_post_df):
        self.logger = logging.getLogger("attack")

        self.attack_model_name = attack_model_name
        self.shadow_post_df = shadow_post_df
        self.target_post_df = target_post_df

        self.attack_model = self.determine_attack_model(attack_model_name)

    def determine_attack_model(self, attack_model_name):
        if attack_model_name == 'LR':
            return LR()
        elif attack_model_name == 'DT':
            return DT()
        elif attack_model_name == 'MLP':
            return MLP()
        elif attack_model_name == 'RF':
            return RF()
        else:
            raise Exception("invalid attack name")

    def train_attack_model(self, feature_construct_method, save_path):
        self.logger.info("training attack model")
        self.shadow_feature = self._concatenate_feature(self.shadow_post_df, feature_construct_method)
        label = self.shadow_post_df.label.astype('int')
        self.attack_model.train_model(self.shadow_feature, label, save_name=save_path + feature_construct_method)

        train_acc = self.attack_model.test_model_acc(self.shadow_feature, label)
        train_auc = self.attack_model.test_model_auc(self.shadow_feature, label)
        self.logger.info("attack model (%s, %s): train_acc: %s | train_auc: %s"
                         % (self.attack_model_name, feature_construct_method, train_acc, train_auc))

        return train_acc, train_auc

    def test_attack_model(self, feature_construct_method):
        self.logger.info("testing attack model")
        self.target_feature = self._concatenate_feature(self.target_post_df, feature_construct_method)
        label = self.target_post_df.label.astype('int')

        test_acc = self.attack_model.test_model_acc(self.target_feature, label)
        test_auc = self.attack_model.test_model_auc(self.target_feature, label)
        self.logger.info("attack model (%s, %s): test_acc: %s | test_auc: %s"
                         % (self.attack_model_name, feature_construct_method, test_acc, test_auc))

        return test_acc, test_auc

    def obtain_attack_posterior(self, post_train, post_test, feature_construct_method):
        self.logger.info("obtaining attack posterior")
        post_train[feature_construct_method] = ""
        post_test[feature_construct_method] = ""

        post = self.attack_model.predict_proba(self.shadow_feature)

        for i in range(post.shape[0]):
            post_train.at[i, feature_construct_method] = post[i]

        post = self.attack_model.predict_proba(self.target_feature)

        for i in range(post.shape[0]):
            post_test.at[i, feature_construct_method] = post[i]

        self.logger.info("obtained attack posterior")

    @staticmethod
    def calculate_comparison_metrics(post_df, feature_construct_method):
        basic_conf, feature_conf = [], []

        for i in range(post_df.shape[0]):
            if post_df.label[i] == 0:
                basic_conf.append(post_df.loc[i, "basic_mia"][0])
                feature_conf.append(post_df.loc[i, feature_construct_method][0])
            elif post_df.label[i] == 1:
                basic_conf.append(post_df.loc[i, "basic_mia"][1])
                feature_conf.append(post_df.loc[i, feature_construct_method][1])
            else:
                raise Exception("invalid label")

        diff = np.array(feature_conf) - np.array(basic_conf)
        improve_rate = np.mean(diff)
        better_rate = np.count_nonzero(diff > 0) / diff.size

        return improve_rate, better_rate

    def _concatenate_feature(self, posterior, method):
        feature = np.zeros((posterior[method][0].shape))

        for _, post in enumerate(posterior[method]):
            feature = np.concatenate((feature, post), axis=0)

        return feature[1:, :]
