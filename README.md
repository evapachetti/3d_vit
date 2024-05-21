# On the effectiveness of 3D vision transformers for the prediction of prostate cancer aggressiveness

Official code for [**On the effectiveness of 3D vision transformers for the prediction of prostate cancer aggressiveness**](https://link.springer.com/chapter/10.1007/978-3-031-13324-4_27) based on [Pytorch reimplementation](https://github.com/jeonsworld/ViT-pytorch) by [jeonsworld](https://github.com/jeonsworld) of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy. 

## Dataset
We utilized the [Prostate-X 2](https://www.cancerimagingarchive.net/collection/prostatex/) dataset for our experiments. To see pre-processing details, please refer to our [paper](https://link.springer.com/chapter/10.1007/978-3-031-13324-4_27).

According to our code, data should be stored according to the following structure:
```
├── dataset
│   └── ProstateX-YYYY
│       ├── original                             
│       ├── rotation
│       ├── horizontal_flip
│       ├── vertical_flip
```
The ProstateX-YYYY folder refers to single patient acquisition, while the four subfolders contain the original and augmented versions of the images.

## Usage

### 1. Train 3D ViTs according to 5-fold cross-validation
```
python train_cv.py
```

## Citation

```bibtex
@inproceedings{10.1007/978-3-031-13324-4_27,
author="Pachetti, Eva
and Colantonio, Sara
and Pascali, Maria Antonietta",
editor="Mazzeo, Pier Luigi
and Frontoni, Emanuele
and Sclaroff, Stan
and Distante, Cosimo",
title="On the Effectiveness of 3D Vision Transformers for the Prediction of Prostate Cancer Aggressiveness",
booktitle="Image Analysis and Processing. ICIAP 2022 Workshops",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="317--328",
abstract="Prostate cancer is the most frequent male neoplasm in European men. To date, the gold standard for determining the aggressiveness of this tumor is the biopsy, an invasive and uncomfortable procedure. Before the biopsy, physicians recommend an investigation by multiparametric magnetic resonance imaging, which may serve the radiologist to gather an initial assessment of the tumor. The study presented in this work aims to investigate the role of Vision Transformers in predicting prostate cancer aggressiveness based only on imaging data. We designed a 3D Vision Transformer able to process volumetric scans, and we optimized it on the ProstateX-2 challenge dataset by training it from scratch. As a term of comparison, we also designed a 3D Convolutional Neural Network, and we optimized it in a similar fashion. The results obtained by our preliminary investigations show that Vision Transformers, even without extensive optimization and customization, can ensure an improved performance with respect to Convolutional Neural Networks and might be comparable with other more fine-tuned solutions.",
isbn="978-3-031-13324-4"
}
```
