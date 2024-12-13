# Model Zoo of CAS-ViT

---

## Classification

| Model | Paras(M) | Top1(%) | Download |
| :--- | :--- | :--- | :--- |
| CAS-ViT-XS | 3.20 | 78.3/78.7 | [Google Drive](https://drive.google.com/file/d/16wKcwF6QMW5w_lyPYnDKjMNuoxQDfrLK/view?usp=drive_link)/[Google Drive](https://drive.google.com/file/d/1kwRPtJ4FdmNeTm2MsiFN7N2DJpou3Pgl/view?usp=drive_link) |
| CAS-ViT-S  | 5.76 | 80.9/81.1 | [Google Drive](https://drive.google.com/file/d/1facFRq8s8oelYUtK1fj3fcfdoWoKDBQQ/view?usp=drive_link)/[Google Drive](https://drive.google.com/file/d/1UagCihMWmNCmYGC1DV5euAvA5TONI818/view?usp=drive_link) |
| CAS-ViT-M  | 12.42 | 82.8/83.0 | [Google Drive](https://drive.google.com/file/d/13sQpSEf0h_uuh0jRy9V0yIW6ZsbDpVGy/view?usp=drive_link)/[Google Drive](https://drive.google.com/file/d/1pTwKKRLPA7vBfk_KTgJ4J1Jq3ttkh3O2/view?usp=drive_link) |
| CAS-ViT-T  | 21.76 | 83.9/84.1 | [Google Drive](https://drive.google.com/file/d/1NqoIUPbwBC91RTjTUvubAbOfGqo1VYT0/view?usp=drive_link)/[Google Drive](https://drive.google.com/file/d/1N5Y81Vcyf2ox41TC3wlRBxgQPYaEndTW/view?usp=drive_link) |



## Object Detection and Instance Segmentation

| Model | Backbone | box AP | mask AP | Config | Download |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RetinaNet | CAS-ViT-XS | 36.5 | - | [Config](./detection/configs/RCViT/retinanet_rcvit_xs_fpn_1x_coco_bs4.py) |[Google Drive](https://drive.google.com/file/d/1n3NKegRLC6wz8EgY-dKWaoPETKumfPlO/view?usp=sharing) |
| RetinaNet | CAS-ViT-S  | 38.6 | - | [Config](./detection/configs/RCViT/retinanet_rcvit_s_fpn_1x_coco_bs4.py) | [Google Drive](https://drive.google.com/file/d/18msm_na7s25aNV7AfBPpJRMGFChpnC5-/view?usp=sharing) |
| RetinaNet | CAS-ViT-M  | 40.9 | - | [Config](./detection/configs/RCViT/retinanet_rcvit_m_fpn_1x_coco_bs4.py) | [Google Drive](https://drive.google.com/file/d/1k5S6by4i6vEmr0P_w35z1ipyTFu59T7z/view?usp=sharing) |
| RetinaNet | CAS-ViT-T  | 41.9 | - | [Config](./detection/configs/RCViT/retinanet_rcvit_t_fpn_1x_coco_bs4.py) | [Google Drive](https://drive.google.com/file/d/1tp6x0pP-6zZvyPGQlvFtMBpw3WpomGac/view?usp=sharing) |
| Mask R-CNN | CAS-ViT-XS | 37.4 | 34.9 | [Config](./detection/configs/RCViT/mask_rcnn_rcvit_xs_fpn_1x_coco_bs4.py) | [Google Drive](https://drive.google.com/file/d/1E8ZhO708_5J7iCmqN1Hc7HH0OAGmlMsH/view?usp=sharing) |
| Mask R-CNN | CAS-ViT-S  | 39.8 | 36.7 | [Config](./detection/configs/RCViT/mask_rcnn_rcvit_s_fpn_1x_coco_bs4.py) | [Google Drive](https://drive.google.com/file/d/1Kttud7Pqc_Tvw--F19knbfF4haJdJOG8/view?usp=sharing) |
| Mask R-CNN | CAS-ViT-M  | 42.3 | 38.9 | [Config](./detection/configs/RCViT/mask_rcnn_rcvit_m_fpn_1x_coco_bs4.py) | [Google Drive](https://drive.google.com/file/d/1zI_N7YCfpi95L6Jr0FLhVIT4u-Y2sidO/view?usp=sharing) |
| Mask R-CNN | CAS-ViT-T  | 43.5 | 39.6 | [Config](./detection/configs/RCViT/mask_rcnn_rcvit_t_fpn_1x_coco_bs4.py) | [Google Drive](https://drive.google.com/file/d/1CKGqkFC763gHEmNtXbalF0YBDN6Dhv7_/view?usp=sharing)|



## Semantic Segmentation

| Model | Backbone | mIoU | Config | Download |
| :--- | :--- | :--- | :--- | :--- |
| Semantic FPN | CAS-ViT-XS | 37.1 | [Config](./segmentation/configs/RCViT/fpn_rcvit_xs_512x512_40k_ade20k_bs4.py) | [Google Drive](https://drive.google.com/file/d/1gCWmdNNQEa9EEwsxIuL2daIavJPcnGny/view?usp=drive_link) |
| Semantic FPN | CAS-ViT-S  | 41.3 | [Config](./segmentation/configs/RCViT/fpn_rcvit_s_512x512_40k_ade20k_bs4.py) | [Google Drive](https://drive.google.com/file/d/1fRY5RZXF7inaDgGKQhLKPpexz5t9p7w2/view?usp=drive_link) |
| Semantic FPN | CAS-ViT-M  | 43.6 | [Config](./segmentation/configs/RCViT/fpn_rcvit_m_512x512_40k_ade20k_bs4.py) | [Google Drive](https://drive.google.com/file/d/1QgBkh006WXqOrw23ehWOHPZM_V_O84c6/view?usp=sharing) |
| Semantic FPN | CAS-ViT-T  | 45.0 | [Config](./segmentation/configs/RCViT/fpn_rcvit_t_512x512_40k_ade20k_bs4.py) | [Google Drive](https://drive.google.com/file/d/14tn6hctKu16GKNBeUiwuUzJml-qsxdkh/view?usp=sharing) |
