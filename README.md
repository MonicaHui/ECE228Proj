# ECE228Proj: A reinforcement cartoon style transfer method
In this project, we proposed multiple strategies to reinforce the cartoon style migration on real-world photos. Based on [CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf) , attention masks for foreground and backgroundwere fused to the generator. 

Besides, additional losses were introduced for image texture and color enhancement. Kernel decomposition was also applied in the residual blocks for computation and storage optimization.

## Prepare Dataset

```
data
└── real
    └── train [ real-world images ]
        └── subfolder
            └── img folder
└── cartoon
    ├── train [ cartoon-style images ]
        └── subfolder
            └── img folder
    └──  edge_smoothed [ produced by util.edge_smooth ]
        └── subfolder
            └── img folder
```

## Train
```
python main.py \
    --lr_G 0.0001 \
    --lr_D 0.0001 \
    --cont_lambda 10 \
    --gray_lambda 10 \
    --adv_lambda 5 \
    --add_attention True \
    --add_losses True
```
use default setting to run our model, train model without added losses by setting ```add_losses``` to False, train model with original structure by setting ```add_attention``` to False

## Reference
reference from:\
https://github.com/JackyWang2001/CartoonGAN_pytorch\
https://github.com/MonicaHui/Decompose-Distill-BeautyGAN\
https://github.com/Ha0Tang/AttentionGAN\
https://github.com/TachibanaYoshino/AnimeGAN

