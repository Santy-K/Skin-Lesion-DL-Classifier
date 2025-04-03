import torch
import pandas as pd 
import os 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

# Data set: 254 seb, 1372 normal, 374 melanoma
# 0 - neither melanoma nor seb 
# 1 - melanoma
# 2 - seb 

##CONSTANTS 
MELANOMA = 1
NEITHER = 0
SEB = 2 

# Data preprocessing variables 
image_width = 256
image_height = 256 


"""
I had to seperate the images from superpixel images in the training data file using
mv ISIC_*******.png images

Add the address for the ground truth csv file on your computer here - we should do this more neatly later 
but it was taking wayyy too long to just upload the dataset with the code to git and I took wayyy too long
playing with different ideas of how to do so. 
"""

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

transforms = transforms.Compose([
    transforms.Resize((image_width, image_height)),
    transforms.ToTensor()
    #Save the images in a file
])

"""
Loads the data for the 2017 data set. We can add transforms for the targets and images later.
Note that I proabably should have put the modifications to the ground truth csv file in here 
but I didnt think of doing that when I was coding it. """
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

# Testing that have correctly loaded data
train_img, train_label = train_dataset[1]
test_img, test_label = test_dataset[1]
valid_img, valid_label = valid_dataset[1]
#img.show()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyper parameters 
num_epochs = 10
num_classes = 3 
batch_size = 50 
learning_rate = 0.001 

#Data Loader 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=batch_size, 
                                             shuffle=False)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                batch_size=batch_size, 
                                                shuffle=False)
 
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
        self.fc = nn.Linear(64*64*32, num_classes)
        
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

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    total_correct = 0
    total_total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #if (i+1) % 100 == 0:
        #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")
        
        #Check accuracy
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        total_total += total
        total_correct += correct
    print(f'Accuracy of the network on the {total_total} train images: {100 * total_correct // total_total} %')
    
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
        

#Validate the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network on the {total} validation images: {100 * correct // total} %')
        
