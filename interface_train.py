import questionary
from prepare_data import main as preprocess_ufpr
from prepare_data_cityscapes import main as preprocess_cityscapes
import logging
from datetime import datetime
from train import train_function
import os
import yaml

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
ufpr_preprocessed_path = "/home/ahsan/sam_finetune/SAM-fine-tune/ufpr_dataset"
cityscapes_preprocessed_path = "/home/ahsan/sam_finetune/SAM-fine-tune/cityscapes_dataset"

# Default configurations for training
default_train_config = {
    'TRAIN_PATH': './dataset/training',
    'TEST_PATH': './dataset/testing',
    'IMAGE_FORMAT': '.png',
    'BATCH_SIZE': 1,
    'NUM_EPOCHS': 15,
    'LEARNING_RATE': 0.0001,
    'PATIENCE': 10
}

default_sam_config = {
    'CHECKPOINT': './sam_vit_h_4b8939.pth',
    'RANK': "512"
}

# Select dataset (UFPR or Cityscapes)
dataset_name = questionary.select(
    "Select your data?",
    default="UFPR",
    choices=["UFPR", "Cityscapes"]
).ask()

# Ask for paths and preprocessing options
if dataset_name == "UFPR":
    data_root_path = questionary.path(
        default="/tmp/ahsan/sqfs/storage_local/datasets/public/ufpr-alpr",
        message=f"Please select your root data location for {dataset_name}",
        only_directories=True
    ).ask()

    preprocessed_check = questionary.select(
        "Have you already preprocessed the data?",
        choices=["Yes", "No"]
    ).ask()

    if preprocessed_check == "No":
        logger.info("Starting UFPR data preprocessing...")
        preprocess_ufpr(dataset_path=data_root_path, split="training")
        logger.info("UFPR training data preprocessing completed.")
        preprocess_ufpr(dataset_path=data_root_path, split="testing")
        logger.info("UFPR testing data preprocessing completed.")
    else:
        logger.info("Using preprocessed UFPR data.")

elif dataset_name == "Cityscapes":
    data_root_path = questionary.path(
        default="/tmp/ahsan/sqfs/storage_local/datasets/public/cityscapes",
        message=f"Please select your root data location for {dataset_name}",
        only_directories=True
    ).ask()

    preprocessed_check = questionary.select(
        "Have you already preprocessed the data?",
        choices=["Yes", "No"]
    ).ask()

    if preprocessed_check == "No":
        logger.info("Starting Cityscapes data preprocessing...")
        preprocess_cityscapes(dataset_path=data_root_path, split="train")
        logger.info("Cityscapes training data preprocessing completed.")
        preprocess_cityscapes(dataset_path=data_root_path, split="test")
        logger.info("Cityscapes testing data preprocessing completed.")
    else:
        logger.info("Using preprocessed Cityscapes data.")

# Gather training parameters from the user
training_parameters = questionary.form(
    rank=questionary.select(
        "Select the rank (default 512):",
        choices=["2", "4", "6", "8", "16", "32", "64", "128", "256", "512"],
        default=default_sam_config['RANK']
    ),
    epochs=questionary.text(
        "Enter desired number of epochs (recommended 10-20):",
        default=str(default_train_config['NUM_EPOCHS'])
    ),
    model_type=questionary.select(
        "Select Model Type:",
        choices=['SAM-Base', 'SAM-Huge']
    )
).ask()

# Define paths depending on the dataset
if dataset_name == "UFPR":
    train_path = os.path.join(ufpr_preprocessed_path, "training")
    test_path = os.path.join(ufpr_preprocessed_path, "testing")
else:
    train_path = os.path.join(cityscapes_preprocessed_path, "training")
    test_path = os.path.join(cityscapes_preprocessed_path, "testing")

# Final configuration file with user inputs and default values
final_config = {
    'DATASET': {
        'TRAIN_PATH': train_path,
        'TEST_PATH': test_path,
        'IMAGE_FORMAT': default_train_config['IMAGE_FORMAT'],
        'TYPE':dataset_name
    },
    'SAM': {
        'CHECKPOINT': default_sam_config['CHECKPOINT'],
        'RANK': int(training_parameters['rank'])
    },
    'TRAIN': {
        'BATCH_SIZE': default_train_config['BATCH_SIZE'],
        'NUM_EPOCHS': int(training_parameters['epochs']),
        'LEARNING_RATE': default_train_config['LEARNING_RATE'],
        'PATIENCE': default_train_config['PATIENCE'], 
        "RESUME_TRAINING":True
    },
    "MODEL":{
        "TYPE":"vit-h" if 'vit_h' in default_sam_config['CHECKPOINT'] else 'vit-b'
    }
}

# Save the configuration file with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config_filename = f"{timestamp}_config.yaml"
config_path = os.path.join(os.getcwd(), config_filename)

with open(config_path, 'w') as config_file:
    yaml.dump(final_config, config_file)

logger.info(f"Config file saved as {config_filename}")

# Start the training process with the generated config file
train_function(config_path)

logger.info("Training Started")
