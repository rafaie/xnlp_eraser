local MODEL_NAME ="bert-base-uncased";

{
    "random_seed": 12345,
    "numpy_seed": 12345,
    "pytorch_seed": 12345,
    "dataset_reader": {
        "type": "single_doc_reader",
        "ds_name": "e-snli",
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": MODEL_NAME,
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": MODEL_NAME,
        },
        'labels': ['contradiction', 'entailment', 'neutral'],
    },
    "data_loader": {
        "type": "multiprocess",
        "batch_size": 32,
    },
    "train_data_path": "./data/esnli/val.jsonl",
    "validation_data_path": "./data/esnli/val.jsonl",
    "test_data_path": "./data/esnli/test.jsonl",
    "evaluate_on_test": true,
    "model": {
        "type": "fine_tune_baseline",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": MODEL_NAME
                }
            }
        }
    },
    "trainer": {
        "num_epochs": 10,
        "patience": 3,
        "cuda_device": 0,
        "grad_clipping": 5.0,
        "grad_norm": 1.0,
        "validation_metric": "-loss",
        "optimizer": {
            "type": "adam",
            "lr": 1e-5
        },
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 10,
            "num_steps_per_epoch": 97957
        }
    }
}