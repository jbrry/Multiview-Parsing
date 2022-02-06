//local transformer_model = std.extVar("MODEL_NAME");
local transformer_model = "xlm-roberta-base";
local max_length = 128;
local transformer_dim = 768;
local encoder_dim = transformer_dim;

local batch_size = 8;

// the pretrained transformer is common to all readers
local reader_common = {
  "multi_token_indexers": {
    "tokens": {
      "type": "pretrained_transformer_mismatched",
      "model_name": transformer_model,
      "max_length": max_length,
      "tokenizer_kwargs": {
        "do_lower_case": false,
        "tokenize_chinese_chars": true, // true for mbert
        "strip_accents": false,
        "clean_text": true,
      }
    }
  }
};

{
  "dataset_reader": {
    "type": "multitask",
    "readers": {
      "TBID_PLACEHOLDER": reader_common {
        "type": "universal_dependencies_multi",
        "mono_sentence_character_token_indexers" : {
          "token_characters" : {
            "type": "single_id",
            "namespace": "TBID_PLACEHOLDER"
          }
        }
      }
    }
  },
  "train_data_path": {
    "TBID_PLACEHOLDER": "_",
  },
  "validation_data_path": {
    "TBID_PLACEHOLDER": "_",
  },
  "test_data_path": {
    "TBID_PLACEHOLDER": "_",
  },
  "model": {
    "type": "multitask_v2",
    "multiple_heads_one_data_source": true,
    "desired_order_of_heads" : ["multi_dependencies", "mono_dependencies", "meta_dependencies"],
    "backbone": {
      "type": "multiview",
        "multi_embedder": {
          "token_embedders": {          
            "tokens": {
              "type": "pretrained_transformer_mismatched",
              "model_name": transformer_model,
              "max_length": max_length
            }
          }
        },
        "mono_sentence_character_embedders": {      
          "TBID_PLACEHOLDER": {
            "token_embedders" : {
              "tokens": {
                "type": "embedding",
                "vocab_namespace": "PLACEHOLDER_tokens",
                "embedding_dim": 25,
              }
            }
          }
        },
        "mono_encoders": {
          "TBID_PLACEHOLDER": {
            "type": "stacked_bidirectional_lstm",
            "input_size": 25,
            "hidden_size": 100,
            "num_layers": 2,
            "recurrent_dropout_probability": 0.3,
            "use_highway": true
          }
        },
    "dropout": 0.33,
    "input_dropout_word": 0.33,
    //"input_dropout_character": 0.05
    },
    "heads": {
      "multi_dependencies": {
        "type": "multiview_multi_parser",
        "encoder_dim": encoder_dim,
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33
      },
      "mono_dependencies": {
        "type": "multiview_mono_parser",
        "encoder_dim": 200,
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33
      },
      "meta_dependencies": {
        "type": "multiview_meta_parser",
        "meta_encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": transformer_dim + 200,
          "hidden_size": 400,
          "num_layers": 2,
          "recurrent_dropout_probability": 0.33,
          "use_highway": true
        },
        //"encoder_dim": encoder_dim + 200,
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33
      }
    }
  },

  "data_loader": {
    "type": "multitask",
    "scheduler": {
      "batch_size": batch_size
    },
    "shuffle": true,
  },

  "trainer": {
    "num_epochs": 50,
    "grad_norm": 5.0,
    "patience": 10,
    "cuda_device": 0,
    "validation_metric": "+meta_dependencies_LAS",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-4,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
  },
  "evaluate_on_test": true,
  //"random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
  //"numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
  //"pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
}
