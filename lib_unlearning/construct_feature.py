import copy
import numpy as np
from scipy.spatial import distance


class ConstructFeature:
    def __init__(self, posterior_df):
        self.posterior_df = posterior_df

    @staticmethod
    def launch_topk_defense(post_df, top_k):
        # Use top1, top2, top3 posterior, the others use average of `1-sum(posterior values)`
        post_df = copy.deepcopy(post_df)

        for index, posterior in enumerate(post_df.original):
            sort_indices = np.argsort(posterior[0, :])
            sum = 0
            for k in range(top_k):
                sum += posterior[0, sort_indices[-(k+1)]]
            ave = (1-sum) / (sort_indices.size - top_k)

            for i in range(sort_indices.size):
                if i in range(top_k):
                    post_df.original[index][0][sort_indices[-(i+1)]] = posterior[0, sort_indices[-(i+1)]]
                else:
                    post_df.original[index][0][sort_indices[-(i+1)]] = ave

        for index, posterior in enumerate(post_df.unlearning):
            sort_indices = np.argsort(posterior[0, :])
            sum = 0
            for k in range(top_k):
                sum += posterior[0, sort_indices[-(k + 1)]]
            ave = (1 - sum) / (sort_indices.size - top_k)

            for i in range(sort_indices.size):
                if i in range(top_k):
                    post_df.unlearning[index][0][sort_indices[-(i+1)]] = posterior[0, sort_indices[-(i+1)]]
                else:
                    post_df.unlearning[index][0][sort_indices[-(i+1)]] = ave
        return post_df

    @staticmethod
    def launch_label_defense(post_df):
        post_df = copy.deepcopy(post_df)

        for index, posterior in enumerate(post_df.original):
            sort_indices = np.argsort(posterior[0, :])
            for i in range(sort_indices.size):
                if i == sort_indices[-1]:
                    post_df.original[index][0][i] = 1
                else:
                    post_df.original[index][0][i] = 0

        for index, posterior in enumerate(post_df.unlearning):
            sort_indices = np.argsort(posterior[0, :])
            for i in range(sort_indices.size):
                if i == sort_indices[-1]:
                    post_df.unlearning[index][0][i] = 1
                else:
                    post_df.unlearning[index][0][i] = 0

        return post_df

    def obtain_feature(self, method, post_df):
        post_df = copy.deepcopy(post_df)
        self.posterior_df[method] = ""

        if method == "direct_diff":
            self.posterior_df[method] = post_df.original - post_df.unlearning

        elif method == "sorted_diff":
            for index, posterior in enumerate(post_df.original):
                sort_indices = np.argsort(posterior[0, :])
                post_df.original[index] = posterior[0, sort_indices].reshape((1, sort_indices.size))
                post_df.unlearning[index] = post_df.unlearning[index][0, sort_indices].reshape((1, sort_indices.size))
            self.posterior_df[method] = post_df.original - post_df.unlearning

        elif method == "l2_distance":
            for index in range(post_df.shape[0]):
                original_posterior = post_df.original[index][0]
                unlearning_posterior = post_df.unlearning[index][0]
                euclidean = distance.euclidean(original_posterior, unlearning_posterior)
                self.posterior_df[method][index] = np.full((1, 1), euclidean)

        elif method == "direct_concat":
            for index in range(post_df.shape[0]):
                original_posterior = self.posterior_df.original[index]
                unlearning_posterior = self.posterior_df.unlearning[index]
                conc = np.concatenate((original_posterior, unlearning_posterior), axis=1)
                self.posterior_df[method][index] = conc

        elif method == "sorted_concat":
            for index, posterior in enumerate(post_df.original):
                sort_indices = np.argsort(posterior[0, :])
                original_posterior = posterior[0, sort_indices].reshape((1, sort_indices.size))
                unlearning_posterior = post_df.unlearning[index][0, sort_indices].reshape((1, sort_indices.size))
                conc = np.concatenate((original_posterior, unlearning_posterior), axis=1)
                self.posterior_df[method][index] = conc

        elif method == "basic_mia":
            self.posterior_df[method] = self.posterior_df.original

        else:
            raise Exception("invalid feature construction method")

        # pickle.dump(self.posterior_df, open(save_dir + "posterior_" + method, 'wb'))
