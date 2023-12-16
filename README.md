# TC-LIF
Two-Compartment Leaky Integrate-and-Fire Model

The source code for paper: **TC-LIF: A Two-Compartment Spiking Neuron Model for Long-term Sequential Modelling, AAAI 2024**

To view our paper, please refer: [TC-LIF](https://arxiv.org/abs/2308.13250)

## Method
![image](https://github.com/ZhangShimin1/TC-LIF/blob/main/figs/method.png)

## Dependency
The denpendencies and versions are listed below:
```
python                  3.8.13
CUDA                    11.7.1
torch                   1.13.0
torchaudio              0.13.0
torchvision             0.14.0
numpy                   1.23.3
urllib3                 1.26.12
spikingjelly            0.0.0.0.13
librosa                 0.7.1
Werkzeug                2.0.3
```

## Reproduce
### S-MNIST
```
# recurrent: 64-256-256-10 paras: 155.1K
python MNIST/mn_train.py --neuron tclif --beta1 1.5 --beta2 -0.5 --task SMNIST --threshold 1.0 --gamma 0.5 --sg triangle --network fb --ind 1
# feedforward: 64-256-256-10 paras: 63.6K
python MNIST/mn_train.py --neuron tclif --beta1 1.5 --beta2 -0.5 --task SMNIST --threshold 1.0 --gamma 0.5 --sg triangle --network ff --ind 1
```
### PS-MNIST
```
# recurrent: 64-256-256-10 paras: 155.1K
python MNIST/mn_train.py --neuron tclif --beta1 -1.4 --beta2 1.4 --task PSMNIST --threshold 1.8 --gamma 1.0 --sg triangle --network fb --ind 1
# feedforward: 64-256-256-10 paras: 63.6K
python MNIST/mn_train.py --neuron tclif --beta1 0. --beta2 0. --task PSMNIST --threshold 1.5 --gamma 0.7 --sg triangle --network ff --ind 1
```
### GSC
```
# recurrent: 40-300-300-12 paras: 196.5K
python GSC/gg12_train.py --neuron tclif --beta1 1.4 --beta2 1.4 --threshold 1.25 --gamma 0.7 --sg triangle --network fb --version v2 --drop 0.3
# feedforward: 40-300-300-12 paras: 106.2K
python GSC/gg12_train.py --neuron tclif --beta1 0. --beta2 0. --threshold 1.2 --gamma 0.6 --sg triangle --network ff --version v2 --drop 0.3
```
### SHD
```
# recurrent: 700-128-128-20 paras: 141.8K
python SHD-SSC/main.py --neuron tclif --task SHD --beta1 0. --beta2 0. --threshold 1.5 --gamma 0.5 --sg triangle --network fb 
# feedforward: 700-128-128-20 paras: 108.8K
python SHD-SSC/main.py --neuron tclif --task SHD --beta1 0. --beta2 0. --threshold 1.5 --gamma 0.5 --sg triangle --network ff
```
### SSC
```
# recurrent: 700-128-128-135 paras:?
python SHD-SSC/main.py --neuron tclif --task SSC --beta1 0. --beta2 0. --threshold 1.5 --gamma 0.5 --sg triangle --network fb
# feedforward: 700-128-128-135 paras:?
python SHD-SSC/main.py --neuron tclif --task SSC --beta1 0. --beta2 0. --threshold 1.5 --gamma 0.5 --sg triangle --network ff
```
