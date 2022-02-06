{
  "dataset_reader": {
    "type": "universal_dependencies_baseline",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
  },
  train_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-train.conllu',
  validation_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-dev.conllu',
  "model": {
    "type": "pos_tagger",
    "dropout": 0.5,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 50
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "embedding_dim": 16
            },
            "encoder": {
            "type": "lstm",
            "input_size": 64,
            "hidden_size": 64,
            "num_layers": 3,
            "bidirectional": true
            }
          }
       },
    },
    "encoder": {
        "type": "lstm",
        "input_size": 50 + 128,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": 0.5,
        "bidirectional": true
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
        "num_serialized_models_to_keep": 3,
    },
    "validation_metric": "+f1-measure-overall",
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
  }
}