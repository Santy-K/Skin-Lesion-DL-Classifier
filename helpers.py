import torch
import pandas as pd 
import os 
from PIL import Image
import numpy as np
from torch.amp import autocast
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time
import torchvision.transforms as v2
from collections import Counter
import random
from sklearn.metrics import confusion_matrix
#Fix for h5py sometimes not being able to open files in parallel
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py

class networkTraining():
    """Class for training and testing a neural network model.
    This class handles the training and testing process, including saving the model.
    """
    def __init__(self, model, optimizer, criterion):
        """Creates the network training class.
        This class handles the training and testing process, including saving the model.

        Args:
            model: Model to be trained.
            optimizer: Optimizer to be used.
            criterion: Loss function to be used.
        """
        self.device_type = get_device()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(self.device_type)
        self.model.to(self.device, non_blocking=True)
        self.history = {}

    #Adapted fromkuzu_main.py, hw1 of COMP9444
    def train(self, train_loader: DataLoader, epoch:int=0):
        """Trains the model for one epoch using the given data loader. Epoch is used for logging purposes.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            epoch (int, optional): Epoch of the cycle. Used for logging. Defaults to 0.
        """
        self.model.train()
        correct = 0
        loss_total = 0
        total_images = 0
        
        t = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
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
            
            total_images += len(data)

            print(f"Train Epoch: {epoch} [{total_images}/{len(train_loader.dataset)}]")

        self.history.setdefault(epoch, {})["train_loss"] = loss_total / batch_idx
        self.history[epoch]["train_accuracy"] = correct / len(train_loader.dataset)
        self.history[epoch]["train_time"] = time.time() - t
        print(f"Train Epoch: {epoch}\t Loss: {loss_total / batch_idx:.6f} \t Accuracy: {correct}/{len(train_loader.dataset)} ({correct / len(train_loader.dataset):.2%})")
    
    #Adapted fromkuzu_main.py, hw1 of COMP9444
    def test(self, test_loader: DataLoader, name:str="test", epoch:int=0):
        """Tests the model using the given data loader. Name is used for logging purposes.
        
        Args:
            train_loader (DataLoader): DataLoader for the training data.
            name (str, optional): Name of the test set. Used for logging. Defaults to "Test".
        """
        self.model.eval()

        test_loss = 0
        correct = 0
        total_samples = 0
        batches = len(test_loader)

        t = time.time()
        
        #inference_mode is faster than no_grad
        with torch.inference_mode():
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
        
        self.history.setdefault(epoch, {})[f"{name}_loss"] = test_loss / batches
        self.history[epoch][f"{name}_accuracy"] = correct / len(test_loader.dataset)
        self.history[epoch][f"{name}_time"] = time.time() - t
        
        print(f"\n{name}: Average loss: {test_loss / batches:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({correct / len(test_loader.dataset):.2%}%)\n")
        
        #Confusion matrix
        conf_matrix = confusion_matrix(target.cpu(), pred.cpu())
        print(f"Confusion matrix for {name}:")
        print(conf_matrix)

        
    def save_model(self, path, model_name="history/model"):
        #Save the model
        torch.save(self.model.state_dict(), path)
        
        #Save the history of the model
        df = pd.DataFrame(self.history).T
        df = df.rename_axis('epoch').reset_index()
        #Create the folder if it does not exist
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        df.to_csv(f"{model_name}_history.csv", index=False)
    
    def plot_model(self, path='plot.jpg', model_name="model"):
        df = pd.DataFrame(self.history).T
        df = df.rename_axis("epoch").reset_index()
        
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        
        #Loss and accuracy
        loss_list = [x for x in list(df) if "loss" in x]
        acc_list = [x for x in list(df) if "accuracy" in x]
        
        for loss in loss_list:
            ax[0].plot(df["epoch"], df[loss], label=loss)
        
        for acc in acc_list:
            ax[1].plot(df["epoch"], df[acc], label=acc)
        
        #Labelling and formatting
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        
        ax[1].set_title("Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
        ax[1].set_ylim((0, 1))
        ax[1].legend()
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
       
        #Title
        fig.suptitle(f"{model_name} Performance")
        plt.savefig(path)
        
        plt.show()
    
def get_device():
    """Finds the device to be used for training and testing.
    This is used to determine if a GPU is available.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    
    return 'cpu'
            
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
                transform = v2.Compose([
                    v2.Resize(size=(image_height, image_width)),
                    #v2.CenterCrop(size=(image_height, image_width)),
                ])
                img = transform(img)
                # Convert to NumPy array
                img_np = np.array(img, dtype=np.uint8)

            dset_images[i] = img_np
            dset_labels[i] = label
            
            if i % 200 == 0:
                print(f"Processed {i}/{len(df['truth'])} images")

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
    
def save_transformed_image(tensor, filename, output_dir="transformed_images"):
    #Saves a transformed image tensor to a file.
    os.makedirs(output_dir, exist_ok=True)
    # Add back the mean and std if you want to visualize the original color space
    mean = torch.tensor([0.5, 0.5, 0.5]).to(tensor.device)[:, None, None]
    std = torch.tensor([0.5, 0.5, 0.5]).to(tensor.device)[:, None, None]
    img = tensor * std + mean
    img = img.clamp(0, 1)
    img = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)

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

def print_label_distribution(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        labels = f['labels'][:]
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Label {label}: {count} samples")
    return 0

def count_samples(dataloader):
    labels = np.concatenate([labels.numpy() for _, labels in dataloader])
    return Counter(labels)

def seed_program(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

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

if __name__ == "__main__":
    create_h5_from_images("data/combined_truth.csv", "data/all", "data/all_128.h5", (128, 128))
    #create_h5_from_images("data/combined_truth.csv", "data/all", "data/all_224.h5", (224, 224))
    #create_h5_from_images("data/combined_truth.csv", "data/all", "data/all_240.h5", (240, 240))
    
    print_label_distribution("data/all_128.h5")
    print_label_distribution("data/all_224.h5")
    print_label_distribution("data/all_240.h5")