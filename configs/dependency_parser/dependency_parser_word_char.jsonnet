{

    "random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
    "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
    "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
    "dataset_reader":{
        "type":"universal_dependencies_baseline",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
      },
    "token_characters": {
        "type": "characters",
      }
    }
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("DEV_DATA_PATH"),
    //"train_data_path": "data/ud-treebanks-v2.6/UD_Finnish-FTB/fi_ftb-ud-train.conllu",
    //"validation_data_path": "data/ud-treebanks-v2.6/UD_Finnish-FTB/fi_ftb-ud-dev.conllu",
    "model": {
      "type": "biaffine_parser_baseline",     
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
            "embedding_dim": 64,
            "vocab_namespace": "token_characters"
            },
            "encoder": {
            "type": "lstm",
            "input_size": 64,
            "hidden_size": 64,
            "num_layers": 3,
            "bidirectional": true
            }
          }
        }
      },
      "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": 228,
        "hidden_size": 400,
        "num_layers": 3,
        "recurrent_dropout_probability": 0.33,
        "use_highway": true
      },
      "use_mst_decoding_for_validation": true,
      "arc_representation_dim": 500,
      "tag_representation_dim": 100,
      "dropout": 0.33,
      "input_dropout": 0.33,
      "initializer": {
        "regexes": [
          [".*projection.*weight", {"type": "xavier_uniform"}],
          [".*projection.*bias", {"type": "zero"}],
          [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
          [".*tag_bilinear.*bias", {"type": "zero"}],
          [".*weight_ih.*", {"type": "xavier_uniform"}],
          [".*weight_hh.*", {"type": "orthogonal"}],
          [".*bias_ih.*", {"type": "zero"}],
          [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
      }
    },
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : 32
      }
    },
    "trainer": {
      "num_epochs": 50,
      "grad_norm": 5.0,
      "patience": 50,
      "cuda_device": 0,
      "validation_metric": "+LAS",
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.9],
        "lr": 0.001
      }
    }
}
