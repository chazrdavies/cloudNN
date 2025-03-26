import torchvision
import torch
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import src.SatelliteSegmentationDataset as dataset
import src.models.UNET.unet as unet

from datetime import datetime
#import datetime
import sys

import utils.config as config





def train_one_epoch(model, data_loader, epoch_idx, tb_writer, criterion, optimizer):


    last_loss = 0.0
    for i, data in enumerate(data_loader):
        
        inputs, labels = data

        
        inputs = inputs.to(device)   # Move inputs to the same device as the model
        labels = labels.to(device)   # Move labels to the same device as the model


        
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()


        last_loss = loss.item()
        

        print('  batch {} loss: {}'.format(i + 1, last_loss))
        tb_x = epoch_idx * len(data_loader) + i + 1
        tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    

    return last_loss
        

def train(model, data_loader, val_loader, num_epochs, criterion, optimizer):


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    writer = SummaryWriter('runs/model{}'.format(timestamp))


    epoch_number = 0

    best_vloss = 1_000_000

    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # set model to train
        model.train(True)

        avg_loss = train_one_epoch(model, data_loader, epoch_number, writer, criterion=criterion, optimizer=optimizer)

        model.eval()

        running_loss = 0.0
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):

                val_inputs, val_labels = val_data
                val_inputs = val_inputs.to(device)   # Move inputs to the same device as the model
                val_labels = val_labels.to(device)   # Move labels to the same device as the model

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                running_loss += val_loss

        avg_val_loss = running_loss/ (i+1)


        print('LOSS train {} valid {}'.format(avg_loss, avg_val_loss))

        epoch_number += 1


    writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_val_loss },
                        epoch_number + 1)
    writer.flush()
    


    if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)



        
# for loading in saved models

def load_mini_unet_model(path):

    model = unet.MiniUnet()
    model.load_state_dict(torch.load('path'))

    return model
        



if __name__ == "__main__":


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    model_name="MiniUnet"

    print(f"Running {model_name}...")



    model = unet.MiniUnet(in_channels=config.RGB_NIR_CHANNELS, num_classes=config.NUM_CLASSES).to(device)
    
    for param in model.parameters():
        if param.dim() > 1:  
            torch.nn.init.kaiming_normal_(param)

    train_loader, val_loader = dataset.make_data_loader(config.IMAGE_DIR, config.MASK_DIR, batch_size=config.BATCH_SIZE, train_ratio=config.TRAIN_SPLIT)
    
    print("Data loaders built")
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=config.MOMENTUM)
    criterion = torch.nn.CrossEntropyLoss()

    start_t = datetime.now().time()


    train(model,train_loader,val_loader,num_epochs=config.NUM_EPOCHS, criterion= criterion, optimizer= optimizer)


    print(f"Training took {((datetime.now().time() - start_t)/60.0):.2f} minutes")

    model_path = "results/mini_unet_v1.pth"

    torch.save(model.state_dict(), model_path)
    print(f"Model state dictionary saved to {model_path}")