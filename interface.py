import questionary
from prepare_data import main as preprocess_ufpr
import logging
from datetime import datetime
from train import train_function
import os
logger = logging.getLogger(__name__)
ufpr_preprocessed_path = "/home/ahsan/sam_finetune/SAM-fine-tune/ufpr_dataset"
need_preprocessing = False
default_train_config = {
    'TRAIN_PATH': './ufpr_dataset/training',
    'TEST_PATH': './ufpr_dataset/testing',
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

# Select data
data = questionary.select(
    "Select your data?",
    default="UFPR",
    choices=["UFPR", "City Scapes"]
).ask()

if data == "UFPR":
    data_root_path = questionary.path(
        default="/tmp/ahsan/sqfs/storage_local/datasets/public/ufpr-alpr",
        message=f"Please select your root data location for {data}",
        only_directories=True
    ).ask()

    preprocessed_check = questionary.select(
        "Have you already preprocessed the data?",
        choices=["Yes", "No"]
    ).ask()

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

    if preprocessed_check == "Yes":
        preprocessed_path = questionary.path(
            default=ufpr_preprocessed_path,
            message=f"Please select your preprocessed data location for {data}",
            only_directories=True
        ).ask()
    else:
        need_preprocessing = True

    if need_preprocessing:
        logger.info("Please wait while your dataset is preprocessing ...")
        preprocess_ufpr(dataset_path=data_root_path, split="training")
        logger.info("Done preprocessing training set")
        logger.info("Now preprocessing testing set")
        preprocess_ufpr(dataset_path=data_root_path, split="testing")
        logger.info("Preprocessing Done...")

    logging.info("Now Starting the training")

    # Save configuration
    final_config = {
        'DATASET': {
            'TRAIN_PATH': default_train_config['TRAIN_PATH'],
            'TEST_PATH': default_train_config['TEST_PATH'],
            'IMAGE_FORMAT': default_train_config['IMAGE_FORMAT']
        },
        'SAM': {
            'CHECKPOINT': default_sam_config['CHECKPOINT'],
            'RANK': int(training_parameters['rank'])
        },
        'TRAIN': {
            'BATCH_SIZE': default_train_config['BATCH_SIZE'],
            'NUM_EPOCHS': int(training_parameters['epochs']),
            'LEARNING_RATE': default_train_config['LEARNING_RATE'],
            'PATIENCE': default_train_config['PATIENCE']
        }
    }

    # Here you can save this final_config to a config file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path=os.path.join(os.getcwd(),f"{timestamp}_config.yaml")
    with open(file_path, 'w') as config_file:
        import yaml
        yaml.dump(final_config, config_file)

    logging.info("Config file saved as config.yaml")
    train_function(file_path)
    logging.info("Training Started")
