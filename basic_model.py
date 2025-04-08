import torch
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from helpers import HDF5Dataset, networkTraining
from models import ConvNetModel_1

def main():
    #Set the seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    
    #0.1 Data preprocessing variables 
    image_width = 128
    image_height = 128 
    train_h5 = "data/train.h5"
    test_h5  = "data/test.h5"
    valid_h5 = "data/valid.h5"

    #0.2 Transforms
    transforms_2017 = v2.Compose([
        v2.Resize((image_width, image_height)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        v2.ToTensor()
    ])

    transforms_train = v2.Compose([
        v2.RandomResizedCrop(scale=(0.8, 1.0), size=(image_width, image_height),ratio=(0.9, 1.1)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        v2.Resize((image_width, image_height)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        #v2.GaussianNoise(mean=0.0, sigma=0.2),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        v2.ToTensor()
    ])

    
    #1. Dataset loading
    train_dataset = HDF5Dataset(train_h5, transform=transforms_train)
    valid_dataset = HDF5Dataset(valid_h5, transform=transforms_2017)
    test_dataset  = HDF5Dataset(test_h5,  transform=transforms_2017)
    
    #Check the size of the data set
    print("Train size: ", len(train_dataset))
    print("Test size: ", len(test_dataset))
    print("Validation size: ", len(valid_dataset))

    #Samplers
    unique_labels, counts = np.unique(train_dataset.all_labels, return_counts=True)
    label_to_count = dict(zip(unique_labels, counts))
    sample_weights_train = [1.0 / label_to_count[label]
                            for label in train_dataset.all_labels]

    batch_size = 256
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights_train,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,    # Weighted sampler for train
        shuffle=False,
        num_workers=2,           # More workers for faster data loading
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    #2. Hyperparameters
    learning_rate = 0.001
    num_epochs = 20
    
    #3. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNetModel_1(3).to(device)

    #4. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #5. Training and testing
    trainer = networkTraining(model, optimizer, criterion)
    for epoch in range(num_epochs):
        print("Training epoch: ", epoch)
        trainer.train(train_loader, epoch)
        print("Testing epoch: ", epoch)
        trainer.test(valid_loader, "Validation")

    #6. Final testing/validation
    trainer.test(test_loader, "Test")
    
    #Save the model
    trainer.save_model("model.pth")

if __name__ == "__main__":
    main()
