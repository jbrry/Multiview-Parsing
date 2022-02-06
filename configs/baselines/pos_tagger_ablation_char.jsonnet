// POS tagger with character features
{
  "dataset_reader": {
    "type": "universal_dependencies_baseline",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
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
    "type": "pos_tagger_ablation",
    "dropout": 0.33,
    "character_embedder": {
      "token_embedders" : {
        "token_characters": {
          "type": "embedding",
          "embedding_dim": 32
        }
      }
    },
    "character_encoder": {
      "type": "lstm",
      "input_size": 32,
      "hidden_size": 64,
      "bidirectional": true
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 128,
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
    "patience": 6,
  }
}