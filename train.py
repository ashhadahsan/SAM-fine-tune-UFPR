import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import yaml
from torch.utils.tensorboard import SummaryWriter
import time

import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
from src.lora import LoRA_sam
import os
os.environ['CUDA_VISIBLE_DEVICES']='9'

# Load the config file
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Take dataset path
train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]

# Load SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])

# Create SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
model = sam_lora.sam

# Process the dataset
processor = Samprocessor(model)
train_ds = DatasetSegmentation(config_file, processor, mode="train")

# Create a dataloader
train_dataloader = DataLoader(train_ds, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)

# Initialize optimizer and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=config_file["TRAIN"]["LEARNING_RATE"], weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set model to train and into the device
model.train()
model.to(device)

# Track loss for plotting
total_loss = []

# Initialize SummaryWriter
writer = SummaryWriter()

# Early stopping parameters
patience = config_file["TRAIN"]["PATIENCE"]
best_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Start the timer
start_time = time.time()

for epoch in range(num_epochs):
    epoch_losses = []
    torch.cuda.empty_cache()

    for i, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(batched_input=batch, multimask_output=False)
        stk_gt, stk_out = utils.stacking_batch(batch, outputs)
        stk_out = stk_out.squeeze(1)
        stk_gt = stk_gt.unsqueeze(1)  # We need to get the [B, C, H, W] starting from [H, W]
        loss = seg_loss(stk_out, stk_gt.float().to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    mean_loss = mean(epoch_losses)
    total_loss.append(mean_loss)
    print(f'EPOCH: {epoch}')
    print(f'Mean loss training: {mean_loss}')

    # Log the mean loss for the epoch
    writer.add_scalar('Loss/train', mean_loss, epoch)

    # Check if we need to update the best model
    if mean_loss < best_loss:
        best_loss = mean_loss
        epochs_no_improve = 0
        # torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print('Early stopping triggered.')
        early_stop = True
        break

if not early_stop:
    print('Training completed without early stopping.')

# End the timer
end_time = time.time()
total_training_time = end_time - start_time

# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]

sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")

# Close the writer
writer.close()

# Plot the epoch vs. loss graph
plt.figure(figsize=(10, 5))
plt.plot(range(len(total_loss)), total_loss, marker='o')
plt.title(f'Training Loss per Epoch\nTotal Training Time: {total_training_time:.2f} seconds\n Rank: {rank}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
plt.savefig("loss_epoch_plot.png")
