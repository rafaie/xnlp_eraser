from allennlp.commands import main
import json
import shutil
import sys
import os
import argparse
import subprocess

from datetime import datetime

from sklearn import metrics

os.environ["TOKENIZERS_PARALLELISM"] = "true"


DEFAULT_DATASET = 'cose'
DEFAULT_EXPERIMENT = 'experiment_3'
DEFAULT_DATASET_PATH = os.path.join(os.getcwd(), 'data')
DEFAULT_OUTPUT_PATH = 'output'
DEFAULT_SEED = 0
DEFAULT_TRAIN_FILE = 'train.jsonl'
DEFAULT_DEV_FILE = 'val.jsonl'
DEFAULT_TEST_FILE = 'test.jsonl'
DEFAULT_CONFIG = "experiment_3/conf/base_2docs_model.jsonnet"
DEFAULT_CUDA_DEVICE = 0
DEFAULT_BATCH_SIZE = 32
DEFAULT_PREDICT_BATCH_SIZE = 4
DEFAULT_EXP_NAME = 'test_1'
DEFAULT_RS_WEIGHT = 1
DEFAULT_CLASSIFIER = 'base_2docs_model'
DEFAULT_LOSS_MODE = 'all'
DEFAULT_ATTENTION_CLASS = 'CrossModality'


def train(dataset, config_file, dataset_path, experiment,
          train_file, dev_file, test_file, output_path, seed, exp_name,
          cuda_device, batch_size, rs_weight, loss_mode, att_class):
    dataset_folder = os.path.join(dataset_path, dataset)  # "./data/cose/"
    train_data_path = os.path.join(
        dataset_folder, train_file)  # "./data/cose/val.jsonl"
    dev_data_path = os.path.join(
        dataset_folder, dev_file)  # "./data/cose/val.jsonl"
    test_data_path = os.path.join(
        dataset_folder, test_file)  # "./data/cose/test.jsonl"
    now = datetime.now()
    config_name = os.path.basename(config_file).split('.')[0]
    dt = now.strftime("%Y_%m_%d_%H_%M_%S")
    # "experiment_3/output/cose/"
    output_base_path = os.path.join(
        experiment, output_path, dataset, config_name, dt)

    os.environ["data_base_path"] = dataset_folder
    os.environ["CONFIG_FILE"] = config_file
    os.environ["TRAIN_DATA_PATH"] = train_data_path
    os.environ["DEV_DATA_PATH"] = dev_data_path
    os.environ["TEST_DATA_PATH"] = test_data_path
    os.environ["OUTPUT_BASE_PATH"] = output_base_path
    os.environ["CUDA_DEVICE"] = str(cuda_device)
    os.environ["SEED"] = str(seed)
    os.environ["BATCH_SIZE"] = str(batch_size)
    os.environ["exp_name"] = exp_name
    os.environ["rs_weight"] = str(rs_weight)
    os.environ["LOSS_MODE"] = str(loss_mode)
    os.environ["ATT_CLASS"] = str(att_class)

    # Use overrides to train on CPU.
    # overrides = json.dumps({"trainer": {"cuda_device": str(cuda_device)}})

    # Assemble the command into sys.argv
    run_training = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", output_base_path,
        "--include-package", experiment,
        # "-o", overrides,
    ]

    # sys.argv = export_cmd + ["&&"] + run_training
    sys.argv = run_training

    main()

    return dt, output_base_path

def find_lr(dataset, config_file, dataset_path, experiment,
          train_file, dev_file, test_file, output_path, seed, exp_name,
          cuda_device, batch_size, rs_weight, loss_mode):
    dataset_folder = os.path.join(dataset_path, dataset)  # "./data/cose/"
    train_data_path = os.path.join(
        dataset_folder, train_file)  # "./data/cose/val.jsonl"
    dev_data_path = os.path.join(
        dataset_folder, dev_file)  # "./data/cose/val.jsonl"
    test_data_path = os.path.join(
        dataset_folder, test_file)  # "./data/cose/test.jsonl"
    now = datetime.now()
    config_name = os.path.basename(config_file).split('.')[0]
    dt = now.strftime("%Y_%m_%d_%H_%M_%S")
    # "experiment_3/output/cose/"
    output_base_path = os.path.join(
        experiment, output_path, dataset, config_name, dt)

    os.environ["data_base_path"] = dataset_folder
    os.environ["CONFIG_FILE"] = config_file
    os.environ["TRAIN_DATA_PATH"] = train_data_path
    os.environ["DEV_DATA_PATH"] = dev_data_path
    os.environ["TEST_DATA_PATH"] = test_data_path
    os.environ["OUTPUT_BASE_PATH"] = output_base_path
    os.environ["CUDA_DEVICE"] = str(cuda_device)
    os.environ["SEED"] = str(seed)
    os.environ["BATCH_SIZE"] = str(batch_size)
    os.environ["exp_name"] = exp_name
    os.environ["rs_weight"] = str(rs_weight)
    os.environ["LOSS_MODE"] = str(loss_mode)

    # Use overrides to train on CPU.
    # overrides = json.dumps({"trainer": {"cuda_device": str(cuda_device)}})

    # Assemble the command into sys.argv
    run_training = [
        "allennlp",  # command name, not used by main
        "find-lr",
        config_file,
        "-s", output_base_path,
        "--include-package", experiment,
        "--num-batches", "500",
        # "-o", overrides,
    ]

    # sys.argv = export_cmd + ["&&"] + run_training
    sys.argv = run_training

    main()

    return dt, output_base_path


def predict(dataset, dataset_path, data_file, experiment,
            classifier, output_dir, exp_name, batch_size,
            cuda_device, att_class):
    dataset_folder = os.path.join(dataset_path, dataset)  # "./data/cose/"
    # "./data/cose/val.jsonl"
    data_path = os.path.join(dataset_folder, data_file)
    output_base_path = output_dir

    os.environ["data_base_path"] = dataset_folder
    os.environ["dataset_name"] = dataset
    os.environ["classifier"] = classifier
    os.environ["output_dir"] = output_dir
    os.environ["exp_name"] = exp_name
    os.environ["BATCH_SIZE"] = str(batch_size)
    os.environ["CUDA_DEVICE"] = str(cuda_device)
    os.environ["ATT_CLASS"] = str(att_class)

    os.environ["TEST_DATA_PATH"] = data_path

    # Assemble the command into sys.argv
    run_predict = [
        "allennlp",  # command name, not used by main
        "predict",
        "--output-file", os.path.join(output_dir,
                                      data_file.split('.')[0]+'_pred.jsonl'),
        "--predictor", "rationale_predictor",
        "--include-package",  experiment,
        "--silent",
        "--cuda-device", str(cuda_device),
        "--use-dataset-reader",
        "--dataset-reader-choice", "validation",
        os.path.join(output_dir, "model.tar.gz"), data_path
    ]

    sys.argv = run_predict
    print(run_predict)

    main()


def gen_metric(dataset, dataset_path, data_file, output_dir):
    dataset_folder = os.path.join(dataset_path, dataset)  # "./data/cose/"
    output_base_path = output_dir

    metrics = [
        "python3",
        "eraserbenchmark/rationale_benchmark/metrics.py",
        "--data_dir", dataset_folder,
        "--split", data_file,
        "--results", os.path.join(output_dir,
                                  data_file.split('.')[0]+'_pred.jsonl'),
        "--score_file", os.path.join(output_dir,
                                     data_file.split('.')[0]+'_score.json')
    ]

    print(metrics)
    subprocess.run(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset",
                        dest="dataset",
                        type=str,
                        default=DEFAULT_DATASET,
                        help="Dataset Name. Example: coes")

    parser.add_argument("-c", "--config",
                        dest="config",
                        type=str,
                        default=DEFAULT_CONFIG,
                        help="Config file path.")

    parser.add_argument("-p", "--dataset_path",
                        dest="dataset_path",
                        type=str,
                        default=DEFAULT_DATASET_PATH,
                        help="Dataset path.")

    parser.add_argument("-e", "--experiment",
                        dest="experiment",
                        type=str,
                        default=DEFAULT_EXPERIMENT,
                        help="experiment name.")

    parser.add_argument("--train_file",
                        dest="train_file",
                        type=str,
                        default=DEFAULT_TRAIN_FILE,
                        help="train file name. default: train.jsonl")

    parser.add_argument("--dev_file",
                        dest="dev_file",
                        type=str,
                        default=DEFAULT_DEV_FILE,
                        help="dev file name. default: dev.jsonl")

    parser.add_argument("--test_file",
                        dest="test_file",
                        type=str,
                        default=DEFAULT_TEST_FILE,
                        help="test file name. default: test.jsonl")

    parser.add_argument("--output_path",
                        dest="output_path",
                        type=str,
                        default=DEFAULT_OUTPUT_PATH,
                        help="output path. default: output")

    parser.add_argument("--seed",
                        dest="seed",
                        type=int,
                        default=DEFAULT_SEED,
                        help="seed number. default: 0")

    parser.add_argument("--exp_name",
                        dest="exp_name",
                        type=str,
                        default=DEFAULT_EXP_NAME,
                        help="exp_name. default: test_1")

    parser.add_argument("--classifier",
                        dest="classifier",
                        type=str,
                        default=DEFAULT_CLASSIFIER,
                        help="classifier. default: bert_encoder_generator")

    parser.add_argument("--pred_output_path",
                        dest="pred_output_path",
                        type=str,
                        help="pred_output_path.")

    parser.add_argument("--cuda_device",
                        dest="cuda_device",
                        type=int,
                        default=DEFAULT_CUDA_DEVICE,
                        help="cuda_device. default: 0")

    parser.add_argument("--batch_size",
                        dest="batch_size",
                        type=int,
                        default=DEFAULT_BATCH_SIZE,
                        help="batch_size. default: 32")

    parser.add_argument("--predict_batch_size",
                        dest="predict_batch_size",
                        type=int,
                        default=DEFAULT_PREDICT_BATCH_SIZE,
                        help="predict_batch_size. default: 4")

    parser.add_argument("--rs_weight",
                        dest="rs_weight",
                        type=int,
                        default=DEFAULT_RS_WEIGHT,
                        help="rs_weight. default: 1")

    parser.add_argument("--train_only",
                        dest="train_only",
                        action='store_true',
                        help="train_only. default: False")

    parser.add_argument("--predict_only",
                        dest="predict_only",
                        action='store_true',
                        help="predict_only. default: False")

    parser.add_argument("--loss_mode",
                        dest="loss_mode",
                        type=str,
                        default=DEFAULT_LOSS_MODE,
                        help="loss_mode. default: all")

    parser.add_argument("--find_lr",
                        dest="find_lr",
                        action='store_true',
                        help="find_lr. default: False")

    parser.add_argument("--att_class",
                        dest="att_class",
                        default=DEFAULT_ATTENTION_CLASS,
                        help="att_class")

    args = parser.parse_args()
    output_base_path = None

    if args.find_lr is True:
        find_lr(dataset=args.dataset,
            config_file=args.config,
            dataset_path=args.dataset_path,
            experiment=args.experiment,
            train_file=args.train_file,
            dev_file=args.dev_file,
            test_file=args.test_file,
            output_path=args.output_path,
            seed=args.seed,
            exp_name=args.exp_name,
            cuda_device=args.cuda_device,
            batch_size=args.batch_size,
            rs_weight=args.rs_weight,
            loss_mode=args.loss_mode)
        quit()

    if args.predict_only is False:
        dt, output_base_path = train(dataset=args.dataset,
                                     config_file=args.config,
                                     dataset_path=args.dataset_path,
                                     experiment=args.experiment,
                                     train_file=args.train_file,
                                     dev_file=args.dev_file,
                                     test_file=args.test_file,
                                     output_path=args.output_path,
                                     seed=args.seed,
                                     exp_name=args.exp_name,
                                     cuda_device=args.cuda_device,
                                     batch_size=args.batch_size,
                                     rs_weight=args.rs_weight,
                                     loss_mode=args.loss_mode,
                                     att_class=args.att_class)

    if args.train_only is False:
        if output_base_path is None:
            output_base_path = args.pred_output_path

        predict(dataset=args.dataset,
                dataset_path=args.dataset_path,
                data_file=args.train_file,
                experiment=args.experiment,
                classifier=args.classifier,
                output_dir=output_base_path,
                exp_name=args.exp_name,
                batch_size=args.predict_batch_size,
                cuda_device=args.cuda_device,
                att_class=args.att_class)

        gen_metric(dataset=args.dataset,
                   dataset_path=args.dataset_path,
                   data_file='train',
                   output_dir=output_base_path)

        predict(dataset=args.dataset,
                dataset_path=args.dataset_path,
                data_file=args.dev_file,
                experiment=args.experiment,
                classifier=args.classifier,
                output_dir=output_base_path,
                exp_name=args.exp_name,
                batch_size=args.predict_batch_size,
                cuda_device=args.cuda_device,
                att_class=args.att_class)

        gen_metric(dataset=args.dataset,
                   dataset_path=args.dataset_path,
                   data_file='val',
                   output_dir=output_base_path)

        predict(dataset=args.dataset,
                dataset_path=args.dataset_path,
                data_file=args.test_file,
                experiment=args.experiment,
                classifier=args.classifier,
                output_dir=output_base_path,
                exp_name=args.exp_name,
                batch_size=args.predict_batch_size,
                cuda_device=args.cuda_device,
                att_class=args.att_class)

        gen_metric(dataset=args.dataset,
                   dataset_path=args.dataset_path,
                   data_file='test',
                   output_dir=output_base_path)
