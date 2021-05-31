# DS-Net

Code accompanying the paper  
***Learning Diverse-Structured Networks for Adversarial Robustness*** [paper](https://arxiv.org/abs/2102.01886)  
<!-- -->


## Training and Evaluation

**On CIFAR-10 and SVHN:**

```
python train_ds_net.py --gpu x --start_evaluation 0 --note xxx --factor 4 --init_channel 20 --trades_beta 6 --is_trades 0 --is_mart 0 --is_at 0 --epsilon 0.031 --is_softmax 1 --is_normal 1 --use_amp 1 --epochs 120 --step_size 0.007 --seed xx
```
+ Use "is_trades", "is_at" and "is_mart" to switch among three adversarial training styles.

+ Use "start_evaluation" to specify which epoch you would like to start evaluation.

+ Use "is_softmax" to determine whether the attention weights should be learned.

+ Use "is_normal" and so on to select the initialization for the attention weights.

+ Use "use_amp" to determine whether you would like to use mixed precision training. To use mixed-precision training, follow the apex installation instructions [here](https://github.com/NVIDIA/apex#quick-start)

+ Adversarial training detailes can be selected by changing "trades_beta", "epsilon", "epochs", "step_size", etc.

+ DS-Net detailes can be selected by changing "factor" and "init_channel".

## Citation
If you use any part of this code in your research, please cite our [paper](https://arxiv.org/abs/2102.01886):
```
@article{du2021dsnet,
  title={Learning Diverse-Structured Networks for Adversarial Robustness},
  author={Du, Xuefeng and Zhang, Jingfeng and Han, Bo and Liu, Tongliang and Rong, Yu and Niu, Gang and Huang, Junzhou and Sugiyama, Masashi},
  journal={ICML},
  year={2021}
}
```