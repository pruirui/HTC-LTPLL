# Long-tailed Partial Label Learning by Head Classifier and Tail Classifier Cooperation

This is a official [PyTorch](http://pytorch.org) implementation for **Long-tailed Partial Label Learning by Head Classifier and Tail Classifier Cooperation (AAAI24)**. 


## Data Preparation
### CIFAR and SUN397
For CIFAR and SUN397 datasets, One can directly run shell codes and the dataset will be automatically downloaded.
### PASCAL VOC 2007
Download the LT-PLL version of PASCAL VOC 2007 and extract it to "./data". This LT-PLL dataset is builded by [RECORDS](https://github.com/MediaBrain-SJTU/RECORDS-LTPLL). 



We provide the following shell codes for model training.
## Run CIFAR10
```shell

python -u train.py  --dataset cifar10 --partial_rate 0.5 --imb_ratio 100 \
--exp-dir experiment/CIFAR10 --data_dir ./data \
--epochs 800 --batch-size 256 --lr 0.01 --wd 1e-3 \
--t 2 --save_ckpt

```

## Run CIFAR100
```shell
python -u train.py  --dataset cifar100 --partial_rate 0.05 --imb_ratio 20 \
--exp-dir experiment/CIFAR100 --data_dir ./data \
--epochs 800 --batch-size 256 --lr 0.01 --wd 1e-3 \
--t 2 --save_ckpt
```

## Run SUN397
```shell
python -u train.py  --dataset sun397 --partial_rate 0.05 --imb_ratio 1 \
--exp-dir experiment/SUN397 --data_dir ./data \
--epochs 200 --batch-size 128 --lr 0.01 --wd 1e-3 \
--t 2 --save_ckpt
```

## Run PASCAL VOC
```shell
python -u train.py  --dataset voc --partial_rate 0 --imb_ratio 1 \
--exp-dir experiment/VOC --data_dir ./data \
--epochs 200 --batch-size 128 --lr 0.01 --wd 1e-3 \
--t 0.99 --save_ckpt
```

