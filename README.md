# Cigarette butts segmentation pipeline

### All experiments are based on [lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework

## Steps to reproduce experiments

##### 1) Data retrieval
Go to the `data` folder and read `README` file. It has Link to download the dataset

You must place downloaded folder (`cig_butts`) under `data` folder

So, you have inside `data` folder `cig_butts` folder, and inside `cig_butts` folder you have `train` and `val` folders

Then, run `prepare_masks.sh` script. It will create masks from coco annotations.

##### 2) Running experiment
Install requirements

Simply `python train.py -c <path_to_experiment_configs> -t <path_to_trainer_configs>`

For example `python train.py -c params/pan_eff_net_b0.yml -t params/trainer.yml`

You can monitor your experiments via tensorboard, `tensorboard --logdir=<./logs/lightning_logs> --port=<6006> --host=<0.0.0.0>`

You may want to change `.yml` files for your experiments (at least, you should check `trainer` file)

##### 3) Results
You can check out `notebooks/validation.ipynb` to see custom 
validation pipeline + `IoU` & `Dice` metrics on validation dataset

##### HOWTO: infer
Check `notebooks/example_inference.ipynb`

Check `notebooks/mil_infer.ipynb` for details, how to generate rle_predictions and html page ([mil](http://machine-intelligence.ru/))