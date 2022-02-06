{
    "random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
    "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
    "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
    "dataset_reader":{
        "type":"universal_dependencies_word_char",
        "word_token_indexer" : {
          "tokens" : {
            "type": "single_id",
            "namespace": "token_vocab"
          }
        },
        "character_token_indexer" : {
          "token_characters" : {
            "type": "single_id",
            "namespace": "character_vocab"
          }
        }
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("DEV_DATA_PATH"),
    //train_data_path: 'data/ud-treebanks-v2.6/UD_Finnish-FTB/fi_ftb-ud-train.conllu',
    //validation_data_path: 'data/ud-treebanks-v2.6/UD_Finnish-FTB/fi_ftb-ud-dev.conllu',

    "model": {
      "type": "multitask_word_char",
      "desired_order_of_heads" : ["character", "word", "meta"],
      "allowed_arguments": {
        "backbone" : ["words", "characters"],
        "word": ["encoded_words", "mask", "lemmas", "upos", "xpos", "feats", "head_indices", "head_tags", "metadata"],
        "character": ["encoded_first_and_last_characters", "mask", "lemmas", "upos", "xpos", "feats", "head_indices", "head_tags", "metadata"],
        "meta": ["encoded_words", "encoded_first_and_last_characters", "mask", "lemmas", "upos", "xpos", "feats", "head_indices", "head_tags", "metadata"]
    },
    "backbone": {
      "type" : "word_char",
        "word_embedder": {
          "token_embedders": {          
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "token_vocab",
            "embedding_dim": 100,
            "pretrained_file": std.extVar("VECS_PATH"),
            "trainable": true,
          }
          }
        },
        "character_embedder": {
          "token_embedders": {
          "token_characters": {
            "type": "embedding",
            "vocab_namespace": "character_vocab",
            "embedding_dim": 64,
          }
          }
        },
        "word_encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": 100,
          "hidden_size": 400,
          "num_layers": 3,
          "recurrent_dropout_probability": 0.33,
          "use_highway": true
        },
        "character_encoder": {
          "type": "lstm",
          "input_size": 64,
          "hidden_size": 64,
          "num_layers": 3,
          "bidirectional": true
        },
        "dropout": 0.33,
        "input_dropout_word": 0.33,
        "input_dropout_character": 0.05
    },
    "heads": {
      "character": {
        "type": "character_parser",
          // char has a second encoder
          "encoder": {
            "type": "stacked_bidirectional_lstm",
            "input_size": 256,
            "hidden_size": 200,
            "num_layers": 2,
            "recurrent_dropout_probability": 0.33,
            "use_highway": true
        },
        "encoder_dim": 256,
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33
      },
      "word": {
        "type": "word_parser",
        "encoder_dim": 800,
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33
      },
      "meta": {
        "type": "meta_parser",
        "meta_encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": 1056,
          "hidden_size": 400,
          "num_layers": 2,
          "recurrent_dropout_probability": 0.33,
          "use_highway": true
        },
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
      },
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
      "validation_metric": "+meta_LAS",
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.9],
        "lr": 0.001
      }
    }
}
