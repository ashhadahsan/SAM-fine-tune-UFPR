import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader

import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, build_sam_vit_h
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load the config file
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Settings from config
train_model = config_file["MODEL"]["TYPE"]
dataset_name = config_file["DATASET"]["TYPE"]
sam_checkpoint_path = config_file["SAM"]["CHECKPOINT"]
rank = config_file["SAM"]["RANK"]
epochs=config_file['TRAIN']['NUM_EPOCHS']

# File paths for LoRA finetuned models
LORA_FINETUNED_MODEL_PATH = os.path.join(os.getcwd(), "lora_weights", f"sam_{train_model}_lora_rank_{rank}_data_{dataset_name}_{epochs}_epochs.safetensors")

device = "cuda" if torch.cuda.is_available() else "cpu"
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# Metrics lists
rank_loss, rank_accuracy, rank_precision, rank_recall = [], [], [], []
total_baseline_loss, total_baseline_accuracy, total_baseline_precision, total_baseline_recall = [], [], [], []

# Load SAM model (Baseline)
with torch.no_grad():
    if train_model == "vit-h":
        sam = build_sam_vit_h(checkpoint=sam_checkpoint_path)
    elif train_model == "vit-b":
        sam = build_sam_vit_b(checkpoint=sam_checkpoint_path)
    else:
        raise ValueError(f"Unsupported model type: {train_model}")
    
    processor = Samprocessor(sam)
    dataset = DatasetSegmentation(config_file, processor, mode="test", dataset=dataset_name)
    test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # Baseline evaluation
    sam.eval()
    sam.to(device)
    
    for i, batch in enumerate(tqdm(test_dataloader)):
        outputs = sam(batched_input=batch, multimask_output=False)
        
        gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0)  # [B, C, H, W]
        loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
        total_baseline_loss.append(loss.item())

        predicted_mask = (outputs[0]["low_res_logits"].sigmoid() > 0.5).float()
        predicted_flat = predicted_mask.view(-1).cpu().numpy()
        ground_truth_flat = gt_mask_tensor.view(-1).cpu().numpy()

        total_baseline_accuracy.append(accuracy_score(ground_truth_flat, predicted_flat))
        total_baseline_precision.append(precision_score(ground_truth_flat, predicted_flat))
        total_baseline_recall.append(recall_score(ground_truth_flat, predicted_flat))

    # Log baseline metrics
    baseline_loss = mean(total_baseline_loss)
    baseline_accuracy = mean(total_baseline_accuracy)
    baseline_precision = mean(total_baseline_precision)
    baseline_recall = mean(total_baseline_recall)
    print(f'Baseline - Loss: {baseline_loss}, Accuracy: {baseline_accuracy}, Precision: {baseline_precision}, Recall: {baseline_recall}')

# Evaluation for fine-tuned LoRA models
    # Load SAM model
if train_model == "vit-h":
    sam = build_sam_vit_h(checkpoint=sam_checkpoint_path)
elif train_model == "vit-b":
    sam = build_sam_vit_b(checkpoint=sam_checkpoint_path)

# Create SAM LoRA
sam_lora = LoRA_sam(sam, rank)
sam_lora.load_lora_parameters(LORA_FINETUNED_MODEL_PATH)  
model = sam_lora.sam

processor = Samprocessor(model)
dataset = DatasetSegmentation(config_file, processor, mode="test", dataset=dataset_name)
test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

# Set model to eval mode and move to device
model.eval()
model.to(device)

total_score, total_accuracy, total_precision, total_recall = [], [], [], []

for i, batch in enumerate(tqdm(test_dataloader)):
    outputs = model(batched_input=batch, multimask_output=False)
    
    gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0)
    loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
    total_score.append(loss.item())

    predicted_mask = (outputs[0]["low_res_logits"].sigmoid() > 0.5).float()
    predicted_flat = predicted_mask.view(-1).cpu().numpy()
    ground_truth_flat = gt_mask_tensor.view(-1).cpu().numpy()

    total_accuracy.append(accuracy_score(ground_truth_flat, predicted_flat))
    total_precision.append(precision_score(ground_truth_flat, predicted_flat))
    total_recall.append(recall_score(ground_truth_flat, predicted_flat))

# Log fine-tuned model metrics
mean_tuned_loss = mean(total_score)
mean_tuned_accuracy = mean(total_accuracy)
mean_total_precision = mean(total_precision)
mean_total_recall = mean(total_recall)

print(f'Rank {rank} - Loss: {mean_tuned_loss}, Accuracy: {mean_tuned_accuracy}, Precision: {mean_total_precision}, Recall: {mean_total_recall}')

# Plotting comparison metrics
metrics_results = {
    "Baseline Loss": baseline_loss,
    "Baseline Accuracy": baseline_accuracy,
    "Baseline Precision": baseline_precision,
    "Baseline Recall": baseline_recall,
    f"Rank {rank} Loss": mean_tuned_loss,
    f"Rank {rank} Accuracy": mean_tuned_accuracy,
    f"Rank {rank} Precision": mean_total_precision,
    f"Rank {rank} Recall": mean_total_recall
}

eval_scores_name = ["Loss", "Accuracy", "Precision", "Recall"]
x = np.arange(len(eval_scores_name))
fig, ax = plt.subplots(layout='constrained')

width = 0.2  # Width of the bars
multiplier = 0

for metric_name, score in metrics_results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, score, width, label=metric_name)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title(f'Baseline vs Rank {rank} Comparison')
ax.set_xticks(x + width, eval_scores_name)
ax.legend(loc=3, ncols=2)
ax.set_ylim(0, 1)

plt.savefig(f"./plots/rank_comparison_metrics_{train_model}_{dataset_name}.jpg")

# Create separate plots for each metric
for metric_name, values in metrics_results.items():
    fig, ax = plt.subplots()
    x = np.arange(2)  # Two bars: one for Baseline and one for Rank 512
    width = 0.35

    ax.bar(x[0], values["Baseline"], width, label="Baseline")
    ax.bar(x[1], values[f"Rank {rank}"], width, label=f"Rank {rank}")

    ax.set_ylabel(metric_name)
    ax.set_title(f'Baseline vs Rank {rank} {metric_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline', f'Rank {rank}'])
    ax.legend()

    plt.savefig(f"./plots/baseline_vs_rank{rank}_{metric_name.lower()}.jpg")
    plt.show()
    plt.close()
