import json
import shutil
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from allennlp.commands import main

config_file = "experiment_1/conf/baseline_linear.jsonnet"
model_name = 'bert-base-uncased'
model_type = 'fine_tune_baseline_rational_to_pred_sp'
train_data_path = "./data/esnli/val.jsonl"
validation_data_path ="./data/esnli/val2.jsonl"
test_data_path = "./data/esnli/test.jsonl"
labels = ['neutral', 'entailment', 'contradiction']
labels_str = ','.join(labels)
seed_number = 0
cuda_device = 0
serialization_dir = "./debugger_train"

os.environ["MODEL_NAME"] = model_name
os.environ["MODEL_TYPE"] = model_type
os.environ["SEED_NUMBER"] = str(seed_number)
os.environ["TRAIN_DATA_PATH"] = validation_data_path
os.environ["VALIDATION_DATA_PATH"] = validation_data_path
os.environ["TEST_DATA_PATH"] = test_data_path
os.environ["LABELS"] = labels_str
os.environ["CUDA_DEVICE"] = str(cuda_device)
os.environ["SERIALIZATION_DIR"] = str(serialization_dir)



# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
run_training = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "experiment_1",
    "-o", overrides,
]

# sys.argv = export_cmd + ["&&"] + run_training
sys.argv =  run_training

main()
