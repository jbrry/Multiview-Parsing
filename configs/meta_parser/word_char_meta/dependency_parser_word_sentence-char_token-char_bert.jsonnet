local transformer_model = "bert-base-multilingual-cased";
//local transformer_model = std.extVar("MODEL_NAME");
local max_length = 128;
local transformer_dim = 768;

{
  //"random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
  //"numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
  //"pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
  "dataset_reader":{
      "type":"universal_dependencies_word_char",
      "word_token_indexer" : {
        "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": transformer_model,
          "max_length": max_length
        }
      },
      "token_character_token_indexer" : {
        "token_characters" : {
        "type": "characters",
        "namespace": "token_character_vocab"
        }
      },         
      "sentence_character_token_indexer" : {
        "token_characters" : {
        "type": "single_id",
        "namespace": "sentence_character_vocab"
        }
      }
    },
    //"train_data_path": std.extVar("TRAIN_DATA_PATH"),
    //"validation_data_path": std.extVar("DEV_DATA_PATH"),
    train_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-train.conllu',
    validation_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-dev.conllu',
    
    "model": {
      "type": "multitask_word_char",
      "desired_order_of_heads" : ["sentence_character", "token_character", "word", "meta"],
      "allowed_arguments": {
        "backbone" : ["words", "token_characters", "sentence_characters"],
        "word": ["encoded_words", "mask", "lemmas", "upos", "xpos", "feats", "head_indices", "head_tags", "metadata"],
        "sentence_character": ["encoded_sentence_characters", "mask", "lemmas", "upos", "xpos", "feats", "head_indices", "head_tags", "metadata"],
        "token_character": ["encoded_token_characters", "mask", "lemmas", "upos", "xpos", "feats", "head_indices", "head_tags", "metadata"],
        "meta": ["encoded_words", "encoded_sentence_characters", "encoded_token_characters", "mask", "lemmas", "upos", "xpos", "feats", "head_indices", "head_tags", "metadata"]
    },
    "backbone": {
      "type" : "word_char",
        "word_embedder": {
          "token_embedders": {          
            "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": transformer_model,
            "max_length": max_length
          }
          }
        },

        "token_character_embedder": {
          "token_embedders": {
          "token_characters": {
            "type": "embedding",
            "vocab_namespace": "token_character_vocab",
            "embedding_dim": 64,
          }
          }
        },

        "sentence_character_embedder": {
          "token_embedders": {
          "token_characters": {
            "type": "embedding",
            "vocab_namespace": "sentence_character_vocab",
            "embedding_dim": 64,
          }
          }
        },

        "word_encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": 768,
          "hidden_size": 400,
          "num_layers": 3,
          "recurrent_dropout_probability": 0.33,
          "use_highway": true
        },

        "token_character_encoder": {
          "type": "lstm",
          "input_size": 64,
          "hidden_size": 64,
          "num_layers": 3,
          "bidirectional": true
        },

        "sentence_character_encoder": {
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
      "sentence_character": {
        "type": "sentence_character_parser",
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

      "token_character": {
        "type": "token_character_parser",
          // char has a second encoder
          "encoder": {
            "type": "stacked_bidirectional_lstm",
            "input_size": 128,
            "hidden_size": 200,
            "num_layers": 2,
            "recurrent_dropout_probability": 0.33,
            "use_highway": true
        },
        "encoder_dim": 128,
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
          "input_size": 1056 + 128,
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
        "batch_size" : 4
      }
    },
    "trainer": {
      "num_epochs": 50,
      "grad_norm": 5.0,
      "patience": 50,
      "cuda_device": 0,
      "validation_metric": "+meta_LAS",
      "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-4,
            "parameter_groups": [
              [[".*transformer.*"], {"lr": 1e-5}]
            ]
          },
    }
}
