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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_config(config_file):
    """Load the YAML config file."""
    with open(config_file, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)
    return config


def load_model(train_model, sam_checkpoint_path):
    """Load the appropriate SAM model."""
    if train_model == "vit-h":
        sam = build_sam_vit_h(checkpoint=sam_checkpoint_path)
    elif train_model == "vit-b":
        sam = build_sam_vit_b(checkpoint=sam_checkpoint_path)
    else:
        raise ValueError(f"Unsupported model type: {train_model}")
    return sam


def evaluate_model(model, dataloader, seg_loss, device):
    """Evaluate the model and return the metrics."""
    model.eval()
    model.to(device)

    total_loss, total_accuracy, total_precision, total_recall = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            outputs = model(batched_input=batch, multimask_output=False)

            gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0)
            loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
            total_loss.append(loss.item())

            predicted_mask = (outputs[0]["low_res_logits"].sigmoid() > 0.5).float()
            predicted_flat = predicted_mask.view(-1).cpu().numpy()
            ground_truth_flat = gt_mask_tensor.view(-1).cpu().numpy()

            total_accuracy.append(accuracy_score(ground_truth_flat, predicted_flat))
            total_precision.append(precision_score(ground_truth_flat, predicted_flat))
            total_recall.append(recall_score(ground_truth_flat, predicted_flat))

    mean_loss = mean(total_loss)
    mean_accuracy = mean(total_accuracy)
    mean_precision = mean(total_precision)
    mean_recall = mean(total_recall)

    return mean_loss, mean_accuracy, mean_precision, mean_recall


def plot_metrics(metrics_results, eval_scores_name, train_model, dataset_name, rank):
    """Plot comparison metrics between baseline and fine-tuned models."""
    x = np.arange(len(eval_scores_name))
    fig, ax = plt.subplots(layout='constrained')

    width = 0.2
    multiplier = 0

    for metric_name, score in metrics_results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=metric_name)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Scores')
    ax.set_title(f'Baseline vs Rank {rank} Comparison')
    ax.set_xticks(x + width, eval_scores_name)
    ax.legend(loc=3, ncols=2)
    ax.set_ylim(0, 1)

    plt.savefig(f"./plots/rank_comparison_metrics_{train_model}_{dataset_name}.jpg")
    plt.close()


def plot_individual_metrics(metrics_results, rank):
    """Plot individual metrics between baseline and fine-tuned model."""
    for metric_name, values in metrics_results.items():
        fig, ax = plt.subplots()
        x = np.arange(2)
        width = 0.35

        ax.bar(x[0], values["Baseline"], width, label="Baseline")
        ax.bar(x[1], values[f"Rank {rank}"], width, label=f"Rank {rank}")

        ax.set_ylabel(metric_name)
        ax.set_title(f'Baseline vs Rank {rank} {metric_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline', f'Rank {rank}'])
        ax.legend()

        plt.savefig(f"./plots/baseline_vs_rank{rank}_{metric_name.lower()}.jpg")
        plt.close()


def inference_eval(config_file):
    """Main function to run inference and evaluation."""
    # Load the config
    config = load_config(config_file)

    # Settings from config
    train_model = config["MODEL"]["TYPE"]
    dataset_name = config["DATASET"]["TYPE"]
    sam_checkpoint_path = config["SAM"]["CHECKPOINT"]
    rank = config["SAM"]["RANK"]
    epochs = config["TRAIN"]["NUM_EPOCHS"]

    # Set file paths for LoRA finetuned models
    lora_finetuned_model_path = os.path.join(os.getcwd(), "lora_weights", f"sam_{train_model}_lora_rank_{rank}_data_{dataset_name}_{epochs}_epochs.safetensors")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # Load SAM model (Baseline)
    sam = load_model(train_model, sam_checkpoint_path)

    processor = Samprocessor(sam)
    dataset = DatasetSegmentation(config, processor, mode="test", dataset=dataset_name)
    test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # Evaluate baseline model
    baseline_loss, baseline_accuracy, baseline_precision, baseline_recall = evaluate_model(sam, test_dataloader, seg_loss, device)

    print(f'Baseline - Loss: {baseline_loss}, Accuracy: {baseline_accuracy}, Precision: {baseline_precision}, Recall: {baseline_recall}')

    # Load SAM LoRA fine-tuned model
    sam_lora = LoRA_sam(sam, rank)
    sam_lora.load_lora_parameters(lora_finetuned_model_path)
    model = sam_lora.sam

    # Evaluate fine-tuned model
    tuned_loss, tuned_accuracy, tuned_precision, tuned_recall = evaluate_model(model, test_dataloader, seg_loss, device)

    print(f'Rank {rank} - Loss: {tuned_loss}, Accuracy: {tuned_accuracy}, Precision: {tuned_precision}, Recall: {tuned_recall}')

    # Plot comparison metrics
    metrics_results = {
        "Baseline Loss": baseline_loss,
        "Baseline Accuracy": baseline_accuracy,
        "Baseline Precision": baseline_precision,
        "Baseline Recall": baseline_recall,
        f"Rank {rank} Loss": tuned_loss,
        f"Rank {rank} Accuracy": tuned_accuracy,
        f"Rank {rank} Precision": tuned_precision,
        f"Rank {rank} Recall": tuned_recall
    }

    eval_scores_name = ["Loss", "Accuracy", "Precision", "Recall"]
    plot_metrics(metrics_results, eval_scores_name, train_model, dataset_name, rank)

    # Plot individual metrics
    plot_individual_metrics(metrics_results, rank)
