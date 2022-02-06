{
  "dataset_reader": {
    "type": "universal_dependencies_baseline",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    }
  },
  train_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-train.conllu',
  validation_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-dev.conllu',
  "model": {
    "type": "pos_tagger",
    "dropout": 0.33,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100
        }
      }
  },
  "encoder": {
    "type": "stacked_bidirectional_lstm",
    "input_size": 100,
    "hidden_size": 200,
    "num_layers": 2,
    "recurrent_dropout_probability": 0.33,
    "use_highway": true
  }
  },
  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : 32
      }
  },
  "trainer": {
    "cuda_device": 0,
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "checkpointer": {
        "num_serialized_models_to_keep": 1,
    },
    "validation_metric": "+accuracy",
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 20,
  }
}