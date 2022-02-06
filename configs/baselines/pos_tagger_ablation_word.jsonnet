// POS tagger with word features
{
  "dataset_reader": {
    "type": "universal_dependencies_baseline",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      }
    }
  },
  //"train_data_path": std.extVar("TRAIN_DATA_PATH"),
  //"validation_data_path": std.extVar("DEV_DATA_PATH"),
  train_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-train.conllu',
  validation_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-dev.conllu',
  "model": {
    "type": "pos_tagger_ablation",
    "dropout": 0.33,
    "word_embedder": {
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
    },
    "feedforward": {
                "activations": "elu",
                "dropout": 0.33,
                "hidden_dims": 200,
                "input_dim": 400,
                "num_layers": 1
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
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 6,
    "validation_metric": "+accuracy",
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "checkpointer": {
        "num_serialized_models_to_keep": 1,
    }

  }
}
