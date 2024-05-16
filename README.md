# CAS-ViT: Convolutional Additive Self-attention Vision Transformers for Efficient Mobile Applications

---
Official Implementation of our proposed method CAS-ViT.

<center class="half">
    <img src="./misc/tokenmixer.png" width=100%/>
</center>

Comparison of diverse self-attention mechanisms. (a) is the classical multi-head self-attention in ViT. (b) is the separable self-attention in MobileViTv2, which reduces the feature metric of a matrix to a vector. (c) is the swift self-attention in SwiftFormer, which achieves efficient feature association only with **Q** and **K**. (d) is proposed convolutional additive self-attention.

<center class="half">
    <img src="./misc/arch.png" width=80%/>
</center>

**Upper:** Illustration of the classification backbone network. Four stages downsample the original image to 1/4, 1/8, 1/16, 1/32 . **Lower:** Block architecture with N_i blocks stacked in each stage.

## Classification

### 1. Requirements

```bash
torch==1.8.0
torchvision==0.9.1
timm==0.5.4
```

### 2. Data Prepare

Download ImageNet-1K dataset.
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── ILSVRC2012_val_00000293.JPEG
│  ├── ILSVRC2012_val_00002138.JPEG
│  ├── ......
```

Load image from `./classification/data/imagenet1k/train.txt`.

### 3. Model Zoo

Please refer [Model Zoo](./model_zoo/README.md)

