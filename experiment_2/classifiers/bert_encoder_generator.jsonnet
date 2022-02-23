local is_cose = if std.findSubstr('cose', std.extVar('TRAIN_DATA_PATH')) == [] then false else true;

{
  dataset_reader : {
    type : "rationale_reader" + (if is_cose then '_cose' else ''),
    token_indexers : {
      bert : {
        type : "pretrained_transformer",
        model_name : "bert-base-uncased",
      },
    },
  },
  validation_dataset_reader: {
    type : "rationale_reader" + (if is_cose then '_cose' else ''),
    token_indexers : {
      bert : {
        type : "pretrained_transformer",
        model_name : "bert-base-uncased",
      },
    },
  },
  data_loader: {
        type: "multiprocess",
        batch_size: 32,
    },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: {
    type: "encoder_generator_rationale_model",
    generator: {
      type: "simple_generator_model",
      text_field_embedder: {
        
        token_embedders: {
          bert: {
            type: "pretrained_transformer",
            model_name: 'bert-base-uncased',
          },
        },
      },
      seq2seq_encoder : {
        type: 'lstm',
        input_size: 768,
        hidden_size: 128,
        num_layers: 1,
        bidirectional: true
      },
      dropout: 0.2,
      feedforward_encoder:{
        type: 'pass_through',
        input_dim: 256
      },
    },
    encoder : {
      type: "encoder_rationale_model" + (if is_cose then '_cose' else ''),
      text_field_embedder: {
        token_embedders: {
          bert: {
            type: "pretrained_transformer",
            model_name: 'bert-base-uncased',
          },
        },
      },
      seq2seq_encoder : {
        type: 'lstm',
        input_size: 768,
        hidden_size: 128,
        num_layers: 1,
        bidirectional: true
      },
      dropout: 0.2,
      attention: {
        type: 'additive',
        vector_dim: 256,
        matrix_dim: 256,
      },
      feedforward_encoder:{
        input_dim: 256,
        num_layers: 1,
        hidden_dims: [128],
        activations: ['relu'],
        dropout: 0.2
      },
    },
    reg_loss_lambda: 0.01,
    reg_loss_mu: 2.0,
    reinforce_loss_weight: 1.0,
    rationale_supervision_loss_weight: std.parseInt(std.extVar('rs_weight'))
  },
  trainer: {
    num_epochs: 20,
    patience: 5,
    grad_norm: 10.0,
    validation_metric: "+accuracy",
    cuda_device: std.parseInt(std.extVar("CUDA_DEVICE")),
    checkpointer: {
            serialization_dir:std.extVar('OUTPUT_BASE_PATH'),
            keep_most_recent_by_count: 2
    },
    optimizer: {
      type: "adam",
      lr: 2e-5
    }
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}
