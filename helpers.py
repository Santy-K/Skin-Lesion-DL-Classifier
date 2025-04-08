import torch
import pandas as pd 
import os 
from PIL import Image
import numpy as np
from torch.amp import autocast, GradScaler


#Fix for h5py sometimes not being able to open files in parallel
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py

class networkTraining():
    """Class for training and testing a neural network model.
    This class handles the training and testing process, including saving the model.
    """
    def __init__(self, model, optimizer, criterion):
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(self.device_type)
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    #Adapted fromkuzu_main.py, hw1 of COMP9444
    def train(self, train_loader, epoch=0):
        self.model.train()
        correct = 0
        loss_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            target = target.long()
            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device_type):
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss_total += loss.item()
            
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]")

        print(f"Train Epoch: {epoch}\t Loss: {loss_total / len(train_loader.dataset):.6f} \t Accuracy: {100. * correct / len(train_loader.dataset):.0f}%")
    
    #Adapted fromkuzu_main.py, hw1 of COMP9444
    def test(self, test_loader, name):
        self.model.eval()
        test_loss = 0
        correct = 0
        total_samples = 0
        batches = len(test_loader)

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                with autocast(self.device_type):
                    outputs = self.model(data)
                    target = target.long()
                    loss = self.criterion(outputs, target)

                test_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                correct += (pred == target).sum().item()
                total_samples += target.size(0)
        
        print(f"\n{name}: Average loss: {test_loss / batches:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
            
MELANOMA = 1
NEITHER = 0
SEB = 2 
def create_h5_from_images(csv_path: str, img_dir: str, h5_filename: str, resolution=(224, 224)):
    """Creates an HDF5 file from a CSV file containing images and their labels.
    Images are stored in uint8 format, where the images are (height, width, channels).
    The labels are stored as int.

    Args:
        csv_path (str): Path to the CSV file containing image IDs and labels in the form of 'image_id,truth'.
        img_dir (str): Relative path to the directory containing the images.
        h5_filename (str): Relative path to the output HDF5 file.
        resolution (tuple, optional): Height and width of output images. Defaults to (224, 224).
    """
    df = pd.read_csv(csv_path)
    num_samples = len(df)

    with h5py.File(h5_filename, 'w', locking=False) as f:
        image_width, image_height = resolution
        #Metadata
        f.attrs['resolution'] = (image_width, image_height)
        f.attrs['num_samples'] = num_samples
        f.attrs['num_classes'] = len(df['truth'].unique())
        
        
        #Dataset of images and labels
        dset_images = f.create_dataset("images",
                                       shape=(num_samples, image_height, image_width, 3),
                                       dtype=np.uint8)

        dset_labels = f.create_dataset("labels", shape=(num_samples,), dtype=np.uint8)

        for i, row in df.iterrows():
            img_id = row['image_id']
            label  = row['truth']  # Labelling
            img_path = os.path.join(img_dir, f"{img_id}.jpg")

            #Resize
            with Image.open(img_path).convert('RGB') as img:
                img = img.resize((image_width, image_height), Image.BILINEAR)
                # Convert to NumPy array
                img_np = np.array(img, dtype=np.uint8)

            dset_images[i] = img_np
            dset_labels[i] = label

    print(f"Created {h5_filename} with {num_samples} samples.")
    
class HDF5Dataset(torch.utils.data.Dataset):
    """HDF5 dataset class for loading images and labels from an HDF5 file.
    """
    def __init__(self, filepath: str, transform:str=None):
        """Create a pytorch dataset from an HDF5 file containing images and labels.
        The images are stored in uint8 format, where the images are (height, width, channels).
        The labels are stored as int.
        The dataset is read-only and the file is closed after reading.

        Args:
            filepath (str): Relative path to the HDF5 file.
            transform (str, optional): Transformation of the dataset if required. Defaults to None.
        """
        super().__init__()
        self.filepath = filepath
        self.transform = transform
        
        with h5py.File(filepath, 'r') as f:
            self.images = f['images'][:]
            self.labels = f['labels'][:]
            self.length = len(self.labels)
            self.all_labels = np.array(f['labels'], dtype=np.uint8)
            try:
                self.resolution = f.attrs['size']
            except KeyError:
                self.resolution = (self.images.shape[1], self.images.shape[2])
        
        self.classes = np.unique(self.all_labels)
        
        self.file = None # Initialize file to None
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        #Allows for each worker to open the file independently
        if self.file is None:
            self.file = h5py.File(self.filepath, 'r')
        
        image = self.file['images'][idx]
        label = self.file['labels'][idx]
        
        image = Image.fromarray(image, 'RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def add_truth_column(raw: str, modified: str):
    """Given a csv file with image_id and labels, add a column for truth values.
    0 - neither melanoma nor seb
    1 - melanoma
    2 - seborrheic keratosis

    Args:
        raw (str): Path to the raw csv file with image_id and labels.
        modified (str): Path to the modified csv file with image_id and truth values.
    """
    
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

def convertdata():
    #Example usage of dataset creation
    add_truth_column("data/train.csv", "data/train_truth_1.csv")
    add_truth_column("data/validation.csv", "data/validation_truth.csv")
    add_truth_column("data/test.csv", "data/test_truth_1.csv")
    create_h5_from_images("data/train_truth_1.csv", "data/train", "data/train.h5", (128, 128))
    create_h5_from_images("data/validation_truth.csv", "data/validation", "data/valid.h5", (128, 128))
    create_h5_from_images("data/test_truth_1.csv", "data/test", "data/test.h5", (128, 128))
    
    #We can see the distribution of the labels in the dataset
    print_label_distribution("data/train.h5")
    print_label_distribution("data/valid.h5")
    print_label_distribution("data/test.h5")

def print_label_distribution(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        labels = f['labels'][:]
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Label {label}: {count} samples")
    return 0

if __name__ == "__main__":
    print("test")
    print_label_distribution("data/train.h5")
    print("validation")
    print_label_distribution("data/valid.h5")
    print("test")
    print_label_distribution("data/test.h5")