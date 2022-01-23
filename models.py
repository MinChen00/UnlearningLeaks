import logging
import joblib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from opacus import PrivacyEngine
from tqdm import tqdm

from utils import DataStore


class DNN:
    def __init__(self, net_name, num_classes=10, args=None):
        self.logger = logging.getLogger("DNN")
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.data_store = DataStore(args)
        self.model = self.determine_net(net_name)

    def determine_net(self, net_name, pretrained=False):
        self.logger.debug("determin_net for %s" % net_name)
        self.in_dim = {
            "location": 168,
            "adult": 14,
            "accident": 29,
            "stl10": 96*96*3,
            "cifar10": 32*32*3,
        }
        in_dim = self.in_dim[self.args['dataset_name']]
        out_dim = self.num_classes
        if net_name == "mlp":
            return MLPTorchNet(in_dim=in_dim, out_dim=out_dim)
        elif net_name == "logistic":
            return LRTorchNet(in_dim=in_dim, out_dim=out_dim)
        elif net_name == "simple_cnn":
            return SimpleCNN()
        elif net_name == "resnet50":
            return models.resnet50(pretrained=pretrained, num_classes=out_dim)
        elif net_name == "densenet":
            return models.densenet121(pretrained=pretrained, num_classes=out_dim)
        else:
            raise Exception("invalid net name")

    def train_model(self, train_loader, test_loader, save_name=None):
        self.model = self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'])
        if self.args['optim'] == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=0)

        if self.args['is_dp_defense']:
            privacy_engine = PrivacyEngine(
                self.model,
                sample_rate=self.args['sample_rate'],
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.args['sigma'],
                max_grad_norm=self.args['max_per_sample_grad_norm'],
                secure_rng=self.args['secure_rng'],
            )
            privacy_engine.attach(optimizer)

        criterion = nn.CrossEntropyLoss()
        losses = []
        run_result = []

        self.model.train()
        for epoch, (data, target) in enumerate(tqdm(train_loader, total=self.args['num_epochs'], position=0)):
            self.logger.debug("model name: %s, | model parameters: %s" % (self.args['original_model'], sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if self.args['is_dp_defense']:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(self.args['delta'])
                self.logger.debug(
                    f"Train Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {self.args['delta']}) for α = {best_alpha}"
                )
            else:
                self.logger.debug(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

            train_acc = self.test_model_acc(train_loader)
            test_acc = self.test_model_acc(test_loader)
            self.logger.debug('epoch %s: train acc %s | test acc %s | ovf %s' % (epoch, train_acc, test_acc, train_acc-test_acc))
            run_result.append([epoch, np.mean(losses), train_acc, test_acc, train_acc-test_acc])

        if save_name:
            torch.save(self.model.state_dict(), save_name)

    def load_model(self, save_name):
        self.model.load_state_dict(torch.load(save_name))

    def predict_proba(self, test_case):
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            logits = self.model(test_case.to(self.device))
            posterior = F.softmax(logits, dim=1)
            return posterior.detach().cpu().numpy()

    def test_model_acc(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).to(self.device)
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()
            return correct / len(test_loader.dataset)


class DT:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class RF:
    def __init__(self, min_samples_leaf=30):
        self.model = RandomForestClassifier(random_state=0, n_estimators=500, min_samples_leaf=min_samples_leaf)

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class MLP:
    def __init__(self):
        self.model = MLPClassifier(early_stopping=True, learning_rate_init=0.01)

    def scaler_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class LR:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=400, multi_class='ovr', n_jobs=1)

    def train_model(self, train_x, train_y, save_name=None):
        self.scaler = preprocessing.StandardScaler().fit(train_x)
        # temperature = 1
        # train_x /= temperature
        self.model.fit(self.scaler.transform(train_x), train_y)
        joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        self.scaler = preprocessing.StandardScaler().fit(test_x)
        return self.model.predict_proba(self.scaler.transform(test_x))

    def test_model_acc(self, test_x, test_y):
        # self.load_model(model)
        pred_y = self.model.predict(self.scaler.transform(test_x))

        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(self.scaler.transform(test_x))
        # return roc_auc_score(test_y, pred_y[:, 1])  # binary class classification AUC
        return roc_auc_score(test_y, pred_y[:, 1], multi_class="ovr", average=None)  # multi-class AUC


class SimpleCNN(nn.Module):
    def __init__(self, in_dim=3, out_dim=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # temperature = 2
        # x /= temperature
        # output = F.log_softmax(x, dim=1)
        return x


class MLPTorchNet(nn.Module):
    def __init__(self, in_dim=168, out_dim=9):
        super(MLPTorchNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # temperature = 4
        # x /= temperature
        # return F.log_softmax(x, dim=1)
        return x


class LRTorchNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LRTorchNet, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs
