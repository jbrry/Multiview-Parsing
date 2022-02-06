{
  "random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
  "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
  "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
  "dataset_reader": {
    "type": "universal_dependencies_baseline",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("DEV_DATA_PATH"),
  //train_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-train.conllu',
  //validation_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-dev.conllu',
  "model": {
    "type": "pos_tagger",
    "dropout": 0.33,
    "task": "xpos",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "pretrained_file": std.extVar("VECS_PATH"),
            "trainable": true,
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "embedding_dim": 32
            },
            "encoder": {
            "type": "lstm",
            "input_size": 32,
            "hidden_size": 64,
            "num_layers": 3,
            "bidirectional": true
            }
          }
       },
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 100 + 128,
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
  },
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
    "num_epochs": 50,
    "grad_norm": 5.0,
    "patience": 25,
  }
}