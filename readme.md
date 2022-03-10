# Unlearning-Leaks

This repository contains the implementation for [When Machine Unlearning Jeopardizes Privacy (CCS 2021)](https://arxiv.org/abs/2005.02205).

To run the code, you need first download the dataset, then train target models and shadow models, in the end, launch the attack in our paper.

### Requirements

```
conda create --name unlearningleaks python=3.9
pip3 install sklearn pandas opacus tqdm psutil
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
conda activate unlearningleaks
```

### Directory tree

```
.
├── LICENSE
├── __init__.py
├── config.py
├── data_prepare.py
├── exp.py
├── lib_unlearning
│   ├── attack.py
│   ├── construct_feature.py
│   └── record_split.py
├── main.py
├── models.py
├── parameter_parser.py
├── readme.md
├── temp_data
│   ├── attack_data
│   ├── attack_models
│   ├── dataset
│   ├── processed_dataset
│   ├── shadow_models
│   ├── split_indices
│   └── target_models
└── utils.py
```

### Data Preparation

- **Location** originally comes from [walk2friends: Inferring Social Links from Mobility Profiles](https://arxiv.org/abs/1708.08221 "walk2friends: Inferring Social Links from Mobility Profiles").
- **Adult** originally comes from [“Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid.”](https://archive.ics.uci.edu/ml/datasets/adult "&quot;Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid&quot;&quot;"), or [Kaggle link](https://www.kaggle.com/wenruliu/adult-income-dataset "Kaggle link").
- **Accident** originally comes from [“Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights.”](https://arxiv.org/abs/1909.09638), or [Kaggle link](https://www.kaggle.com/sobhanmoosavi/us-accidents "Kaggle link").
- Three preprocessed categorical datasets can be found at [Google Drive](https://drive.google.com/drive/folders/1e7h9gna39oOxYx9ZNDbUbX5W--OwyD62?usp=sharing).

### Toy examples

```
###### Step 1: Train Original and Unlearned Models ######
python main.py --exp model_train

###### Step 2: Membership Inference Attack under Different Settings ######

###### UnlearningLeaks in 'Retraining from scratch' ######
python main.py --exp mem_inf --unlearning_method scratch

###### UnlearningLeaks in 'SISA'
python main.py --exp model_train --unlearning_method sisa
python main.py --exp mem_inf --unlearning_method sisa

###### UnlearningLeaks in 'Multiple intermediate versions'
python main.py --exp mem_inf --samples_to_evaluate in_out_multi_version

###### UnlearningLeaks in 'Group Deletion'
python main.py --exp model_train --shadow_unlearning_num 10 --target_unlearning_num 10
python main.py --exp mem_inf --shadow_unlearning_num 10 --target_unlearning_num 10

###### UnlearningLeaks in 'Online Learning'
python main.py --exp model_train --samples_to_evaluate online_learning
python main.py --exp mem_inf --samples_to_evaluate online_learning

###### UnlearningLeaks against 'the remaining samples'
python main.py --exp mem_inf --samples_to_evaluate in_in
```

### Citation

```
@inproceedings{chen2021unlearning,
  author    = {Min Chen and Zhikun Zhang and Tianhao Wang and Michael Backes and Mathias Humbert and Yang Zhang},
  title     = {When Machine Unlearning Jeopardizes Privacy},
  booktitle = {{ACM} {SIGSAC} Conference on Computer and Communications Security (CCS)}
  year      = {2021}
}
```
