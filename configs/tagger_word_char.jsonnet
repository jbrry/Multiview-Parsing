{
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
    //"train_data_path": std.extVar("TRAIN_DATA_PATH"),
    //"validation_data_path": std.extVar("DEV_DATA_PATH"),
    train_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-train.conllu',
    validation_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-dev.conllu',
    //train_data_path: 'tests/data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-train.conllu',
    //validation_data_path: 'tests/data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-dev.conllu',
    
    "model": {
      "type": "multitask_word_char",
      "allowed_arguments": {
        "backbone" : ["words", "characters"],
        "word": ["encoded_words", "mask", "pos_tags", "metadata"],
        "character": ["encoded_first_and_last_characters", "mask", "pos_tags", "metadata"],
        "meta": ["word_encoded_words", "character_encoded_characters", "mask", "pos_tags", "metadata"] // mtl model attaches head name to output dict
    },
    "backbone": {
      "type" : "word_char",
        "word_embedder": {
          "token_embedders": {          
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "token_vocab",
            "embedding_dim": 100,
            "sparse": true
          }
          }
        },
        "character_embedder": {
          "token_embedders": {
          "token_characters": {
            "type": "embedding",
            "vocab_namespace": "character_vocab",
            "embedding_dim": 64,
            "trainable": true,
            "sparse": true
          }
          }
        },
        "word_encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": 100,
          "hidden_size": 200,
          "num_layers": 2,
          "recurrent_dropout_probability": 0.3,
          "use_highway": true
        },
        "character_encoder": {
          "type": "lstm",
          "input_size": 64,
          "hidden_size": 64,
          "num_layers": 3,
          "bidirectional": true
        },
        "dropout": 0.3,
        "input_dropout_word": 0.3,
        "input_dropout_character": 0.05
    },
    "heads": {
      "character": {
        "type": "character_tagger",
            "feedforward": {
                "activations": "elu",
                "dropout": 0.3,
                "hidden_dims": 200,
                "input_dim": 256,
                "num_layers": 1
            },
      //  "encoder": {
      //  "type": "stacked_bidirectional_lstm",
      //  "input_size": 256,
      //  "hidden_size": 200,
      //  "num_layers": 2,
      //  "recurrent_dropout_probability": 0.3,
      //  "use_highway": true
      //  },
        // char has a second encoder
        "encoder_dim": 256,
        "dropout": 0.3
      },
      "word": {
        "type": "word_tagger",
            "feedforward": {
                "activations": "elu",
                "dropout": 0.3,
                "hidden_dims": 200,
                "input_dim": 400,
                "num_layers": 1
            },
        "encoder_dim": 400,
        "dropout": 0.3
      },
      "meta": {
        "type": "meta_tagger",
            "feedforward": {
                "activations": "elu",
                "dropout": 0.3,
                "hidden_dims": 200,
                "input_dim": 400,
                "num_layers": 1
            },
        "meta_encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": 400,
          "hidden_size": 200,
          "num_layers": 1,
          "recurrent_dropout_probability": 0.3,
          "use_highway": true
        },
        "encoder_dim": 400
      }
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
      "validation_metric": "+word_acc",
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.9]
      }
    }
}
