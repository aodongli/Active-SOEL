# Deep Anomaly Detection under Labeling Budget Constraints

Official PyTorch implementation of MHRot in [Deep Anomaly Detection under Labeling Budget Constraints (ICML 2023)](https://arxiv.org/abs/2302.07832). 

## Reproduce the Results

This repo contains the code of experiments with SOEL on image data. The backbone model is MHRot. Please note the code will automatically download datasets into `./data/`.

Please run the command to reproduce the results when $K=20$ and anomaly rate equals 10%:

```
## CIFAR-10 ##
# SOEL
python train_ad.py --dataset=cifar10 --epochs=16 --lr=1e-3 --oe_rank=latent_gauss --foldername=cifar10_SOEL_res/ --oe=True --oe_loss=weighted  --K=20 

# Hybr1
python train_ad.py --dataset=cifar10 --epochs=16 --lr=1e-3 --foldername=cifar10_Hybr1_res/ --query_strategy=hybr1 --K=20

# Hybr2
python train_ad.py --dataset=cifar10 --epochs=16 --lr=1e-3 --foldername=cifar10_Hybr2_res/  --oe_loss=supervise --query_strategy=hybr2 --K=20

## FMNIST ##
# SOEL
python train_ad.py --dataset=fmnist --epochs=3 --lr=1e-4 --oe_rank=training_obj --foldername=fmnist_SOEL_res/ --oe=True --oe_loss=weighted --K=20

# Hybr1
python train_ad.py --dataset=fmnist --epochs=15 --lr=1e-4 --foldername=fmnist_Hybr1_res/  --query_strategy=hybr1 --K=20

# Hybr2
python train_ad.py --dataset=fmnist --epochs=15 --lr=1e-4 --foldername=fmnist_Hybr2_res/ --oe_loss=supervise --query_strategy=hybr2 --K=20

## MedMNIST ##
# SOEL
python train_ad.py --dataset=$1 --epochs=15 --lr=1e-4 --oe_rank=training_obj --foldername=$1_SOEL_res/ --oe=True --oe_loss=weighted --K=20

# hybr1
python train_ad.py --dataset=$1 --epochs=15 --lr=1e-4 --foldername=$1_Hybr1_res/  --query_strategy=hybr1 --K=20

# Hybr2
python train_ad.py --dataset=$1 --epochs=15 --lr=1e-4 --foldername=$1_Hybr2_res/ --oe_loss=supervise --query_strategy=hybr2 --K=20
```

For MedMNIST, `$1` is the placeholder for the following options:`blood`, `organa`, `organc`, `organs`.
