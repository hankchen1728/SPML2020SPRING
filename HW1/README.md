# Security and Privacy of Machine Learning, Spring 2020 HOMEWORK 1
## Usage
First change directory to `src`,
1. Training the model
```=python
python3 train.py --visible_gpus [VISIBLE_GPUS] --ckpt_dir [CKPT_DIR] --lr [Learning_rate] --net_name [Architecture] --epochs [EPOCHS]
```

2. Run FGSM attacker
```=python
python3 fgsm.py --data_dir [EVAL_DATA_DIR] --save_dir [ADV_IMG_DIR] --net_name [Architecture] --ckpt_path [CHECKPOINT_PATH]
```

## References
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar): Neural network architectures implemented with `pytorch` for **Cifar10** dataset
- [data_loader.py](https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb): Data loader for **Cifar10** dataset
