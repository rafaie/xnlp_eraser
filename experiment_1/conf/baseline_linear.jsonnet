local MODEL_NAME = std.extVar('MODEL_NAME');
local SEED_NUMBER = std.extVar('SEED_NUMBER');
local TRAIN_DATA_PATH = std.extVar('TRAIN_DATA_PATH');
local VALIDATION_DATA_PATH = std.extVar('VALIDATION_DATA_PATH');
local TEST_DATA_PATH = std.extVar('TEST_DATA_PATH');
local LABELS = std.extVar('LABELS');
local CUDA_DEVICE = std.extVar('CUDA_DEVICE');
local SERIALIZATION_DIR = std.extVar('SERIALIZATION_DIR');

{
    random_seed: SEED_NUMBER,
    numpy_seed: SEED_NUMBER,
    pytorch_seed: SEED_NUMBER,
    dataset_reader: {
        type: "single_doc_reader",
        token_indexers: {
            bert: {
                type: "pretrained_transformer",
                model_name: MODEL_NAME,
            }
        },
        tokenizer: {
            type: "pretrained_transformer",
            model_name: MODEL_NAME,
        },
        labels: LABELS,
    },
    data_loader: {
        type: "multiprocess",
        batch_size: 32,
    },
    train_data_path: TRAIN_DATA_PATH,
    validation_data_path: VALIDATION_DATA_PATH,
    test_data_path: TEST_DATA_PATH,
    evaluate_on_test: true,
    model: {
        type: "fine_tune_baseline",
        embedder: {
            token_embedders: {
                bert: {
                    type: "pretrained_transformer",
                    model_name: MODEL_NAME
                }
            }
        }
    },
    trainer: {
        num_epochs: 20,
        patience: 5,
        cuda_device: CUDA_DEVICE,
        grad_clipping: 5.0,
        grad_norm: 1.0,
        validation_metric: "-loss",
        checkpointer: {
            serialization_dir:SERIALIZATION_DIR,
            num_serialized_models_to_keep: 1
        },
        optimizer: {
            type: "adam",
            lr: 1e-5
        },
        learning_rate_scheduler: {
            type: "slanted_triangular",
            num_epochs: 10,
            num_steps_per_epoch: 97957
        }
    }
}