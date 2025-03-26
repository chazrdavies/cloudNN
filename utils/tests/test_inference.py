from src.models.UNET.unet import MiniUnet
import src.SatelliteSegmentationDataset 
import utils.config as config
import torch

dataLoader,_ = src.SatelliteSegmentationDataset.make_data_loader(config.IMAGE_DIR, config.MASK_DIR, batch_size=3)

batch = next(iter(dataLoader))  # Get a batch
inputs, labels = batch

test_model = MiniUnet(4, 4)

outputs = test_model(inputs)

# check shapes
print("Input shape:", inputs.shape)
print("Output shape:", outputs.shape)
print("Label shape:", labels.shape)

# check softmax
print("Min output:", outputs.min().item())
print("Max output:", outputs.max().item())

#check loss 

optimizer = torch.optim.SGD(test_model.parameters(), lr=config.LR, momentum=config.MOMENTUM)
criterion = torch.nn.CrossEntropyLoss()

loss = criterion(outputs, labels)

print("Loss:", loss.item())

## train small batch

# for epoch in range(10):  # Small number of epochs
    
#     optimizer.zero_grad()
#     outputs = test_model(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

inputs, labels = next(iter(dataLoader))
outputs = test_model(inputs)
loss = criterion(outputs, labels)
loss.backward()

for name, param in test_model.named_parameters():
    if param.grad is None:
        print(f"⚠️ No gradient for {name}")