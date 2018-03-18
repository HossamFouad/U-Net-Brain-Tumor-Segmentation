# U-Net-Brain-Tumor-Segmentation

This implementation is related to brain tumor segmentation, based on BraTS 2017 dataset using Tensorflow.


It includes the following python files:
[1] gen_modalities.py: generate dataset for flair, T1, T1ce, T2 in order to be stacked together to create input where each sample has the four images
[2] avoid_overfitting.py: Truncate the whole black images to avoid overfitting
[3] data_aug.py: augmentation of dataset by several techniques
[4] preprocess.py: shuffling and parsing data
[5] model.py: architecture which is based on U-Net[1]
[6] train.py: train and test the dataset on single label of tumors, included in BraTS 2017 dataset

The project includes the pretrained model in order to avoid the time consumed in training from start point with random initialization.

References: 
[1] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Proc. Med. Image Comput. Comput.-Assisted Intervention, 2015, pp. 234–241.
