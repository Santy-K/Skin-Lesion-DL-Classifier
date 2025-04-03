COMP9444 Group 7

Lesion segmentation

Place test, training and validation data without the superpixel files in to the relevent /data/... files, copy all of these in to data/all

Data sourced from:
https://challenge.isic-archive.com/data/#2017

Basic Model

10 epochs
Learning rate = 0.03
Accuracy of the network on the 2000 train images: 65 %
Accuracy of the network on the 600 test images: 49 %
Accuracy of the network on the 150 validation images: 54 %

Notes about parameters while training base model
- used weighted random sampler on validation and test as well as training 
- used randomly cropped images? on data - need to look into making sure it does not crop out entire mole 
- 
