local transformer_model = std.extVar("MODEL_NAME");
local max_length = 128;
local transformer_dim = 768;
local encoder_dim = transformer_dim;

local batch_size = 8;

local reader_common = {
  "token_indexers": {
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
        "type": "universal_dependencies_all_features",
      }
    }
  },

  "train_data_path": {
    TBID_PLACEHOLDER: std.extVar("TRAIN_DATA_PATH"),
  },
  "validation_data_path": {
    TBID_PLACEHOLDER: std.extVar("DEV_DATA_PATH"),
  },
  "test_data_path": {
    TBID_PLACEHOLDER: std.extVar("TEST_DATA_PATH"),
  },

  "model": {
    "type": "multitask_v2",
    "multiple_heads_one_data_source": true,
    "desired_order_of_heads" : ["first_dependencies", "last_dependencies", "meta_dependencies"],
    "backbone": {
      "type": "first_last",
        "first_last_embedder": {
          "token_embedders": {          
            "tokens": {
              "type": "pretrained_transformer_mismatched_first_last",
              "model_name": transformer_model,
              "max_length": max_length,
              "sub_token_mode": "first_last",
            }
          }
        },

    "dropout": 0.33,
    "input_dropout_word": 0.33,
    },
    "heads": {
      "first_dependencies": {
        "type": "multiview_parser",
        "encoder_dim": encoder_dim,
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33,
        "encoded_text_source": "first_encoded_text"
      },
      "last_dependencies": {
        "type": "multiview_parser",
        "encoder_dim": encoder_dim,
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33,
        "encoded_text_source": "last_encoded_text"
      },
      "meta_dependencies": {
        "type": "multiview_meta_parser",
        "meta_encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": transformer_dim + transformer_dim,
          "hidden_size": 400,
          "num_layers": 2,
          "recurrent_dropout_probability": 0.33,
          "use_highway": true
        },
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33,
        "first_encoded_text_source": "first_encoded_text",
        "second_encoded_text_source": "last_encoded_text"
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
    "validation_metric": "+meta_dependencies_LAS_AVG",
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
  "random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
  "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
  "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
}
