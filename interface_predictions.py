import questionary
from inference_eval import inference_eval

# Prompt the user for the config file path
config_file_path = questionary.path("Please enter the config file path:").ask()

# Pass the provided file path to the inference_eval function
inference_eval(config_file=config_file_path)
