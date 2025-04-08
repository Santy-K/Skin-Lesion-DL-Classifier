COMP9444 Group 7

Lesion segmentation

Place test, training and validation data without the superpixel files in to the relevent /data/... files, copy all of these in to data/all

run `git lfs install`

Data sourced from:
https://challenge.isic-archive.com/data/#2017

Dependencies:
- numpy
- pytorch
- torchvision
- h5py
- pandas

Basic Model 
- used weighted random sampler on validation and test as well as training 
- 120 by 120 images 
- no grey scale 
- 12 masks layer 1 and 32 masks layer 2 

10 epochs
Learning rate = 0.03
Accuracy of the network on the 2000 train images: 65 %
Accuracy of the network on the 600 test images: 49 %
Accuracy of the network on the 150 validation images: 54 %

Notes 
- When changed to grey scale accuracy went down dramatically - after 10 epochs
accuracy was at 41% on test sample 
- when reduced initial number of masks to 6 train accuracy was 59% and test was 55%

Overall
All times the model was run, there was often a big disparity in accuracy between train and test indicating that the model was memorising the data. After discussing with the tutor we decided to prioritise 
- increasing the data set with data augmentation (cropping, colour jitter, rotation etc)
- transfer learning, to account for our small data set 

Things to consider implementing later 
- ensembling - to further reduce 


