import torch
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from helpers import HDF5Dataset, networkTraining, save_transformed_image, count_samples, seed_program
from models import ConvNetModel_1, AlexNetModel
from sklearn.model_selection import train_test_split
from collections import Counter


def main():
    #Set the seed for reproducibility
    seed_program(seed=1)
    
    #0.1 Data preprocessing variables 
    image_scale = (224, 224)

    #0.2 Transforms
    transforms_2017 = v2.Compose([
        v2.Resize(image_scale),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        v2.ToTensor()
    ])

    transforms_train = v2.Compose([
        v2.RandomResizedCrop(scale=(0.8, 1.0), size=image_scale,ratio=(0.9, 1.1)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        #v2.GaussianNoise(mean=0.0, sigma=0.2),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        v2.ToTensor()
    ])

    batch_size = 256
    
    #Read Data
    data_name = "data/all_224.h5"
    data = HDF5Dataset(data_name)
    print("Data size: ", len(data))
    #Print class breakdown
    class_counts = np.bincount(data.all_labels)
    print("Class counts: ", class_counts)
    
    #Train, test, valid split via stratified sampling
    #2000, 550, 200
    train_idx, test_idx, train_label, _ = train_test_split(
        np.arange(len(data.all_labels)), data.all_labels, stratify=data.all_labels, test_size=0.2, random_state=42
    )
    
    train_idx, valid_idx, _, _ = train_test_split(
        train_idx, train_label, stratify=train_label, test_size=1/11, random_state=42
    )
    
    train_dataset = HDF5Dataset(data_name, transform=transforms_train)
    test_dataset = HDF5Dataset(data_name, transform=transforms_2017)
    valid_dataset = HDF5Dataset(data_name, transform=transforms_2017)
    
    train_data = torch.utils.data.Subset(train_dataset, train_idx)
    test_data = torch.utils.data.Subset(test_dataset, test_idx)
    valid_data = torch.utils.data.Subset(valid_dataset, valid_idx)
    
    class_weights = 1.0 / class_counts
    weights = class_weights[train_data.dataset.all_labels[train_data.indices]]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    #Sizes of the datasets
    print("Train size: ", len(train_data), Counter(train_data.dataset.all_labels[train_data.indices]))
    print("Test size: ", len(test_data), Counter(test_data.dataset.all_labels[test_data.indices]))
    print("Valid size: ", len(valid_data), Counter(valid_data.dataset.all_labels[valid_data.indices]))
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        sampler=sampler,
        shuffle = False,
        num_workers=1,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    #Testing the data loading. This is SLOW, so only run to verify the data loading is correct.
    if 0 == 1:
        print("After loading data:")
        print(f"Train: {count_samples(train_loader)}")
        print(f"Test: {count_samples(test_loader)}")
        print(f"Valid: {count_samples(valid_loader)}")
    
        return 0
    
    #2. Hyperparameters, standard for reference
    learning_rate = 0.001
    num_epochs = 75
    
    #3. Model
    model = AlexNetModel()
    #4. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #5. Training and testing
    trainer = networkTraining(model, optimizer, criterion)
    for epoch in range(num_epochs):
        print("Training epoch: ", epoch)      
        trainer.train(train_loader, epoch)
        
        print("Testing epoch: ", epoch)
        trainer.test(valid_loader, "Validation", epoch)
        
        #Save the model
        trainer.save_model(f"models/alexnet_{epoch}.pth", model_name="history/alexnet")

    #6. Final testing/validation
    trainer.test(test_loader, "Test", num_epochs)
    
    #Save the model after final testing
    trainer.save_model(f"models/alexnet_{num_epochs}.pth", model_name="history/alexnet")
    
    trainer.plot_model(model_name="Alexnet")
    
    return 0

if __name__ == "__main__":
    print("Starting")
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down")
        
