{
    "dataset_reader":{
        "type":"universal_dependencies"
    },
    //"train_data_path": std.extVar("TRAIN_DATA_PATH"),
    //"validation_data_path": std.extVar("DEV_DATA_PATH"),

    "train_data_path": "data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-train.conllu",
    "validation_data_path": "data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-dev.conllu",
    "model": {
      "type": "biaffine_parser",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "sparse": true
          }
        }
      },
      //"pos_tag_embedding":{
      //  "embedding_dim": 100,
      //  "vocab_namespace": "pos",
      //  "sparse": true
      //},
      "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": 100,
        "hidden_size": 200,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.3,
        "use_highway": true
      },
      "use_mst_decoding_for_validation": true,
      "arc_representation_dim": 500,
      "tag_representation_dim": 100,
      "dropout": 0.3,
      "input_dropout": 0.3,
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
        "betas": [0.9, 0.9]
      }
    }
}
