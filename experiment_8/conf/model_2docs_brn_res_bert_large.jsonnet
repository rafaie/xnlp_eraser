{
  dataset_reader : {
    type : "rationale_reader_cose",
    token_indexers : {
      bert : {
        type : "pretrained_transformer",
        model_name : "bert-large-uncased",
      },
    },
  },
  validation_dataset_reader: {
    type : "rationale_reader_cose",
    token_indexers : {
      bert : {
        type : "pretrained_transformer",
        model_name : "bert-large-uncased",
      },
    },
  },
  data_loader: {
        type: "multiprocess",
        // batch_size: std.parseInt(std.extVar('BATCH_SIZE')),
        batch_sampler:{
          type:'bucket',
          batch_size: std.parseInt(std.extVar('BATCH_SIZE')),
          // sorting_keys: ['document'],
        },
    },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: {
    type: "model_2docs_brn",
    transfer_method: 2,
    premise_field_embedder:{
      token_embedders: {
        bert: {
          type: "pretrained_transformer",
          model_name: 'bert-large-uncased',
        },
      },
    },
    query_field_embedder:{
      token_embedders: {
        bert: {
          type: "pretrained_transformer",
          model_name: 'bert-large-uncased',
        },
      },
    },
    rationale_model_params : {
      type: "rationale_2docs",
      premise_seq2seq_encoder : {
        type: 'lstm',
        input_size: 1024,
        hidden_size: 256,
        num_layers: 1,
        bidirectional: true
      },
      query_seq2seq_encoder : {
        type: 'lstm',
        input_size: 1024,
        hidden_size: 256,
        num_layers: 1,
        bidirectional: true
      },
      feedforward_encoder:{
        input_dim: 512,
        num_layers: 1,
        hidden_dims: [128],
        activations: ['relu'],
        dropout: 0.2
      },
      att_class_name: std.extVar('ATT_CLASS'),
      agg_func: std.extVar('AGG_FUNC'),
      dropout: 0.2,
    },
    objective_model_params : {
      type: "objective_2docs",
      premise_seq2seq_encoder : {
        type: 'lstm',
        input_size: 1024,
        hidden_size: 256,
        num_layers: 1,
        bidirectional: true
      },
      query_seq2seq_encoder : {
        type: 'lstm',
        input_size: 1024,
        hidden_size: 256,
        num_layers: 1,
        bidirectional: true
      },
      feedforward_encoder:{
        input_dim: 512,
        num_layers: 1,
        hidden_dims: [128],
        activations: ['relu'],
        dropout: 0.2
      },
      att_class_name: std.extVar('ATT_CLASS'), 
      dropout: 0.2,
    },
  },
  trainer: {
    num_epochs: 20,
    patience: 7,
    grad_norm: 10.0,
    validation_metric: "+accuracy",
    cuda_device: std.parseInt(std.extVar("CUDA_DEVICE")),
    checkpointer: {
            serialization_dir:std.extVar('OUTPUT_BASE_PATH'),
            keep_most_recent_by_count: 2
    },
    optimizer: {
      type: "adam",
      lr: 6e-6
    },
    learning_rate_scheduler: {
      type: "step",
      step_size: 1,
      gamma: 0.92,
    },
    callbacks: [
          {
            type: "tensorboard",
            serialization_dir: std.extVar('OUTPUT_BASE_PATH'),
            should_log_parameter_statistics: true,
            should_log_learning_rate:true
            }
    ]
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}
