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
        "word": ["encoded_words", "mask", "pos_tags", "head_indices", "head_tags", "metadata"],
        "character": ["encoded_first_and_last_characters", "mask", "pos_tags", "head_indices", "head_tags", "metadata"],
        "meta": ["word_head_arc_representation", "character_head_arc_representation", "word_child_arc_representation", "character_child_arc_representation", "word_head_tag_representation", "character_head_tag_representation", "word_child_tag_representation", "character_child_tag_representation", "mask", "pos_tags", "head_indices", "head_tags", "metadata"]
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
        "type": "character_parser",
        "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": 256,
        "hidden_size": 200,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.3,
        "use_highway": true
        },
        // char has a second encoder
        "encoder_dim": 256,
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.3
      },
      "word": {
        "type": "word_parser",
        "encoder_dim": 400,
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.3
      },
      "meta": {
        "type": "meta_parser",
        "meta_encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": 2400,
          "hidden_size": 400,
          "num_layers": 1,
          "recurrent_dropout_probability": 0.3,
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
        "betas": [0.9, 0.9]
      }
    }
}
