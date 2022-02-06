import json
import shutil
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from allennlp.commands import main

dataset_folder = "./data/cose/"
config_file = "experiment_2/classifiers/bert_encoder_generator.jsonnet"
train_data_path ="./data/cose/val.jsonl"
dev_data_path ="./data/cose/val.jsonl"
test_data_path = "./data/cose/test.jsonl"
output_base_path="experiment_2/output/cose/"
seed = 0
cuda_device=-1

batch_size = 4
exp_name="test_1"
rs_weight=1

os.environ["data_base_path"] = dataset_folder
os.environ["CONFIG_FILE"] = config_file
os.environ["TRAIN_DATA_PATH"] = train_data_path
os.environ["DEV_DATA_PATH"] = dev_data_path
os.environ["TEST_DATA_PATH"] = test_data_path
os.environ["OUTPUT_BASE_PATH"] = output_base_path
os.environ["CUDA_DEVICE"] = str(cuda_device)
os.environ["SEED"] = str(seed)
os.environ["batch_size"] = str(batch_size)
os.environ["exp_name"] = exp_name
os.environ["rs_weight"] = str(rs_weight)



# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(output_base_path, ignore_errors=True)

# Assemble the command into sys.argv
run_training = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", output_base_path,
    "--include-package", "experiment_2",
    "-o", overrides,
]

# sys.argv = export_cmd + ["&&"] + run_training
sys.argv =  run_training

main()
