import torch
import pandas as pd 
import os 
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
from helpers import networkTraining

# Data set: 254 seb, 1372 normal, 374 melanoma
# 0 - neither melanoma nor seb 
# 1 - melanoma
# 2 - seb 

torch.manual_seed(0)

##CONSTANTS 
MELANOMA = 1
NEITHER = 0
SEB = 2 

# Data preprocessing variables 
image_width = 128
image_height = 128 

image_file = "data/all"
train_dir = "data/train"
test_dir = "data/test"
valid_dir = "data/validation"
train_truth_file = "data/train_truth.csv"
test_truth_file = "data/test_truth.csv"
valid_truth_file = "data/validation_truth.csv"
modified_truth_file = "data/combined_truth_1.csv"
modified_train_truth_file = "data/train_truth_1.csv"
modified_test_truth_file = "data/test_truth_1.csv"
modified_valid_truth_file = "data/validation_truth.csv"
raw_truth_file = "data/all.csv"

"""
Adds a truth column to the groundTruth csv file. This column contains a 
number in range  [0,2] depending on what classification it is and returns it as 
the Modified csv file. 
"""
def add_truth_column(raw, modified):
    df = pd.read_csv(raw)
    output_file = modified
    print(df.shape)
    df.insert(len(df.columns), "truth", NEITHER)
    for index, row in df.iterrows():
        #intially truth is 0 
        # assign melanoma as 1 
        if row['melanoma'] == 1.0 : 
            df.at[index, 'truth'] = MELANOMA
        # assign seb as 2 
        elif row['seborrheic_keratosis'] == 1.0 :
            df.at[index, 'truth'] = SEB

    df.to_csv(output_file)

#add_truth_column(train_truth_file, modified_train_truth_file)
#add_truth_column(test_truth_file, modified_test_truth_file)
#add_truth_column(valid_truth_file, modified_valid_truth_file)

transforms = v2.Compose([
    v2.Resize((image_width, image_height)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    v2.ToTensor()
    #Save the images in a file
])

#Data loader for 2017 data set 
class DataSet17(Dataset) :
    # annotations_file: csv file containing ground truth for each image 
    #img_dir: directory of skin lesion images 
    def __init__(self, annotations_file, img_dir, transform = transforms, target_transform = None) :
        current_directory = os.getcwd()  # Get the current working directory
        print(f"Current working directory: {current_directory}")
        self.img_dir = img_dir
        print("img_dir:", img_dir)
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform 
        self.classes = self.img_labels['truth'].unique()

    def __getitem__(self, index) :
        img_filename = self.img_labels.iloc[index, self.img_labels.columns.get_loc('image_id')]
        img_filename = img_filename + ".jpg"
        label_name = self.img_labels.iloc[index, self.img_labels.columns.get_loc('truth')]
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label_name
        except Exception as e:
            print(f"Error loading image {img_path} at index {index}: {e}")
            raise e
    def __len__(self) :
        return len(self.img_labels)

#Add your own address for the dataset on your computer here 
#dataset = DataSet17(
#    annotations_file = modified_truth_file,
#    img_dir = image_file)

#Split in to train, test and validation data
train_dataset = DataSet17(
    annotations_file = modified_train_truth_file,
    img_dir = train_dir)
test_dataset = DataSet17(
    annotations_file = modified_test_truth_file,
    img_dir = test_dir)
valid_dataset = DataSet17(
    annotations_file = modified_valid_truth_file,
    img_dir = valid_dir)
#Check the size of the data set
print("Train size: ", len(train_dataset))
print("Test size: ", len(test_dataset))
print("Validation size: ", len(valid_dataset))

#exit(1)

#Random weighted sampling
class_counts_train = train_dataset.img_labels['truth'].value_counts().to_dict()
class_counts_test = test_dataset.img_labels['truth'].value_counts().to_dict()
class_counts_valid = valid_dataset.img_labels['truth'].value_counts().to_dict()

sample_weights_train = [1/class_counts_train[label] for label in train_dataset.img_labels['truth']]
sample_weights_test = [1/class_counts_test[label] for label in test_dataset.img_labels['truth']]
sample_weights_valid = [1/class_counts_valid[label] for label in valid_dataset.img_labels['truth']]

# Create a weighted random sampler for the training set
train_sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights_train, num_samples=len(train_dataset), replacement=True)
test_sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights_test, num_samples=len(test_dataset), replacement=True)
valid_sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights_valid, num_samples=len(valid_dataset), replacement=True)

# Testing that have correctly loaded data
train_img, train_label = train_dataset[1]
test_img, test_label = test_dataset[1]
valid_img, valid_label = valid_dataset[1]
#img.show()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyper parameters 
num_epochs = 30
num_classes = 3 
batch_size = 64
learning_rate = 0.03 

#Data Loader 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size, 
                                        sampler=test_sampler)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                        batch_size=batch_size, 
                                        sampler=valid_sampler)
 
class ConvNetModel_1(nn.Module):
    def __init__(self, num_classes = 3):
        super(ConvNetModel_1, self).__init__()
        self.layer1 = nn.Sequential(
            #Because is RGB in channells = 3 
            nn.Conv2d(3, 12, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(32*32*32, num_classes)
        
    def forward(self, x):
        #print("STARTING :")
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        #print(out.shape)
        #print("END")
        return out

model = ConvNetModel_1(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainer = networkTraining(model, optimizer, criterion)
for epoch in range(num_epochs):
    print("Training epoch: ", epoch + 1)
    trainer.train(train_loader, epoch)
    print("Testing epoch: ", epoch + 1)
    trainer.test(valid_loader)

trainer.test(test_loader)