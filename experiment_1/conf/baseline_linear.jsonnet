local MODEL_NAME = std.extVar('MODEL_NAME');
local MODEL_TYPE = std.extVar('MODEL_TYPE');
local SEED_NUMBER = std.extVar('SEED_NUMBER');
local TRAIN_DATA_PATH = std.extVar('TRAIN_DATA_PATH');
local VALIDATION_DATA_PATH = std.extVar('VALIDATION_DATA_PATH');
local TEST_DATA_PATH = std.extVar('TEST_DATA_PATH');
local LABELS = std.extVar('LABELS');
local CUDA_DEVICE = std.extVar('CUDA_DEVICE');
local SERIALIZATION_DIR = std.extVar('SERIALIZATION_DIR');
local LOSS_CO1 = std.parseInt(std.extVar('LOSS_CO1'));
local LOSS_CO2 = std.parseInt(std.extVar('LOSS_CO2'));
local LOSS_CO3 = std.parseInt(std.extVar('LOSS_CO3'));
local LOSS_B = std.parseInt(std.extVar('LOSS_B'));

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
        type: MODEL_TYPE,
        embedder: {
            token_embedders: {
                bert: {
                    type: "pretrained_transformer",
                    model_name: MODEL_NAME
                }
            }
        },
        loss_co1: LOSS_CO1,
        loss_co2: LOSS_CO2,
        loss_co3: LOSS_CO3,
        loss_b: LOSS_B
    },
    trainer: {
        num_epochs: 20,
        patience: 5,
        cuda_device: std.parseInt(CUDA_DEVICE),
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