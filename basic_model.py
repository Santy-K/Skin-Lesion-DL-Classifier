import torch
import pandas as pd 
import os 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Data set: 254 seb, 1372 normal, 374 melanoma
# 0 - neither melanoma nor seb 
# 1 - melanoma
# 2 - seb 

"""
Adds a truth column to the groundTruth csv file. This column contains a 
number in range  [0,2] depending on what classification it is and returns it as 
the Modified csv file. 
"""
def add_truth_column():
    #add the address for the ground truth csv file on your computer here - we should do this more neatly later 
    #but it was taking wayyy too long to just upload the dataset with the code to git and I took wayyy too long
    #playing with different ideas of how to do so. 
    df = pd.read_csv("/Users/katieoreilly/Desktop/UNSW/COMP9444/ISIC-2017_Training_Data/ISIC-2017_Training_Part3_GroundTruth.csv")
    output_file = "/Users/katieoreilly/Desktop/UNSW/COMP9444/ISIC-2017_Training_Data/ISIC-2017_Training_Part3_ModifiedGroundTruth.csv"
    print(df.shape)
    df.insert(len(df.columns), "truth", 0)
    for index, row in df.iterrows():
        #intially truth is 0 
        # assign melanoma as 1 
        if row['melanoma'] == 1.0 : 
            df.at[index, 'truth'] = 1
        # assign seb as 2 
        elif row['seborrheic_keratosis'] == 1.0 :
            df.at[index, 'truth'] = 2

    df.to_csv(output_file)


"""
Loads the data for the 2017 data set. We can add transforms for the targets and images later.
Note that I proabably should have put the modifications to the ground truth csv file in here 
but I didnt think of doing that when I was coding it. """
#Data loader for 2017 data set 
class DataSet17(Dataset) :
    # annotations_file: csv file containing ground truth for each image 
    #img_dir: directory of skin lesion images 
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None) :
        current_directory = os.getcwd()  # Get the current working directory
        print(f"Current working directory: {current_directory}")
        self.img_dir = img_dir
        print("img_dir:", img_dir)
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform 

    def __getitem__(self, index) :
        img_filename = self.img_labels.iloc[index, self.img_labels.columns.get_loc('image_id')]
        img_filename = img_filename + ".jpg"
        label_name = self.img_labels.iloc[index, self.img_labels.columns.get_loc('truth')]
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
            width, height = image.size
            return image, label_name, height, width
        except Exception as e:
            print(f"Error loading image {img_path} at index {index}: {e}")
            raise e

#Add your own address for the dataset on your computer here 
train_dataset = DataSet17(
    annotations_file = "/Users/katieoreilly/Desktop/UNSW/COMP9444/ISIC-2017_Training_Data/ISIC-2017_Training_Part3_ModifiedGroundTruth.csv",
    img_dir = "/Users/katieoreilly/Desktop/UNSW/COMP9444/ISIC-2017_Training_Data/images")

# Testing that have correctly loaded data
img, label, height, width = train_dataset[33]
print(f"Sample 0: Image height: {height}, Image width: {width}, Label: {label}")
img.show()



#Hyper parameters 
num_epochs = 5 
num_classes = 3 
batch_size = 50 
learning_rate = 0.001 

#Data Loader 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
