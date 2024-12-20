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
import os
import requests
import subprocess
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_h, build_sam_vit_b
from src.lora import LoRA_sam

# Run shell script to select the best GPU
subprocess.run("chmod +x select_best_gpu.sh", shell=True)
subprocess.run("./select_best_gpu.sh", shell=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(device)}")

def download_model(url):
    response = requests.get(url)
    model_name = url.split("/")[-1]
    model_path = os.path.join(os.getcwd(), model_name)
    with open(model_path, "wb") as f:
        f.write(response.content)
    return model_path

def get_model_checkpoint(train_model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        if train_model == "vit-b":
            url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth"
        else:
            url = "https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth"
        return download_model(url=url)
    return checkpoint_path

def train_function(config_file_path):
    # Load the config file
    with open(config_file_path, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    # Extract values from config
    train_model = config_file["MODEL"]["TYPE"]
    checkpoint_path = config_file["SAM"]["CHECKPOINT"]
    dataset_name = config_file["DATASET"]["TYPE"]
    rank = config_file["SAM"]["RANK"]
    num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]
    batch_size = config_file["TRAIN"]["BATCH_SIZE"]
    learning_rate = config_file["TRAIN"]["LEARNING_RATE"]
    patience = config_file["TRAIN"]["PATIENCE"]
    resume_training = config_file["TRAIN"].get("RESUME_TRAINING", False)
    
    # Load SAM model checkpoint once (it remains the same)
    sam_checkpoint_path = get_model_checkpoint(train_model, checkpoint_path)
    if train_model == "vit-b":
        sam = build_sam_vit_b(checkpoint=sam_checkpoint_path)
    else:
        sam = build_sam_vit_h(checkpoint=sam_checkpoint_path)

    # Create SAM LoRA
    sam_lora = LoRA_sam(sam, rank)
    model = sam_lora.sam
    model.to(device)

    # Check if we should resume from saved LoRA parameters
    lora_save_path = os.path.join(os.getcwd(), "lora_weights", f"sam_{train_model}_lora_rank_{rank}_data_{dataset_name}.safetensors")
    if resume_training and os.path.exists(lora_save_path):
        sam_lora.load_lora_parameters(lora_save_path)
        print(f"Resumed training from saved LoRA parameters: {lora_save_path}")
        start_epoch = config_file["TRAIN"].get("LAST_EPOCH", 0) + 1
    else:
        start_epoch = 0

    # Process the dataset
    processor = Samprocessor(model)
    train_ds = DatasetSegmentation(config_file, processor, mode="train", dataset=dataset_name)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize optimizer and loss function
    optimizer = Adam(model.image_encoder.parameters(), lr=learning_rate, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # Track training loss
    total_loss = []
    writer = SummaryWriter()
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Start training
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        torch.cuda.empty_cache()

        for i, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(batched_input=batch, multimask_output=False)
            stk_gt, stk_out = utils.stacking_batch(batch, outputs)
            stk_out = stk_out.squeeze(1)
            stk_gt = stk_gt.unsqueeze(1)
            loss = seg_loss(stk_out, stk_gt.float().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = mean(epoch_losses)
        total_loss.append(mean_loss)
        writer.add_scalar('Loss/train', mean_loss, epoch)

        if mean_loss < best_loss:
            best_loss = mean_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save the LoRA parameters after each epoch
        sam_lora.save_lora_parameters(lora_save_path)
        print(f"Saved LoRA parameters to: {lora_save_path}")

        if epochs_no_improve == patience:
            print('Early stopping triggered.')
            early_stop = True
            break

    if not early_stop:
        print('Training completed without early stopping.')

    # Final processing
    end_time = time.time()
    total_training_time = end_time - start_time
    writer.close()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(total_loss)), total_loss, marker='o')
    plt.title(f'Training Loss per Epoch\nTotal Training Time: {total_training_time:.2f} seconds\n Rank: {rank}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f"loss_epoch_plot_{dataset_name}_{train_model}.png")
    plt.show()
