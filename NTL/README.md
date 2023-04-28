# Deep Anomaly Detection under Labeling Budget Constraints

Official PyTorch implementation of NTL in [Deep Anomaly Detection under Labeling Budget Constraints (ICML 2023)](https://arxiv.org/abs/2302.07832). 

## Reproduce the Results

This repo contains the code of experiments with ALOE on various data types including image data and tabular data.

Please run the command and replace \$# with available options (see below):

```
python Launch_Exps.py --config-file $1 --dataset-name $2  --contamination $3 --query_num $4
```

**config-file:**

- `config_img_active.yml`: image data config files; 
- `config_tab_active.yml`: tabular data config files;

**dataset-name:**

- image data: `cifar10_feat`; `fmnist_feat`; `blood`; `organa`; `organc`; `organs`; `path`; `derma`; `pneumonia`;

- tabular data: `pima`; `satellite`; `breastw`; `ionosphere`;

**contamination:**

- The true contamination ratio of the dataset. The default ratio is 0.1.

**query_num:**

- the querying budget size. 

## How to Use

### Active anomaly detection methods

Active anomaly detection methods are different in their querying strategy and the post-query training strategy. The strategies are specified in the config files with field names `query_method` and `train_method`. Different combinations correspond to either ALOE or different baselines.

The available options are:

```
query_method:
- kmeans
- random
- positive
- pos_random
- pos_diverse
- margin
- mar_diverse

train_method:
- loe_est
- sup
- oc
```

### Image data

* Image data is the last-layer features extracted by a ResNet152 pretrained on ImageNet. All image data are downloaded automatically when extracting their features.

* Run `python Extract_img_features.py` to extract CIFAR-10 and F-MNIST features into `./Data`; run  `python Extract_medmnist_features.py`to extract MedMNIST features into`./Data`.

* Run the experiment for active anomaly detection on image data. For example, to reproduce the results of ALOE on CIFAR-10, run
  
  ```
  python Launch_Exps.py --config-file config_img_active.yml --dataset-name cifar10_feat  --contamination 0.1 --query_num 20
  ```

### Tabular data

* The tabular datasets are downloaded from http://odds.cs.stonybrook.edu/. Please put the data under `./DATA`.

* Run the experiment for active anomaly detection on tabular data. For example, to reproduce experiments on the `satellite` dataset with ALOE, run 
  
  ```
  python Launch_Exps.py --config-file config_tab_active.yml --dataset-name satellite --contamination 0.1 --query_num 10
  ```
