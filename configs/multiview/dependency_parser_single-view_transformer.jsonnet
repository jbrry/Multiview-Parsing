local transformer_model = std.extVar("MODEL_NAME");
local max_length = 128;
local transformer_dim = 768;
local encoder_dim = transformer_dim;

local batch_size = 8;

// the pretrained transformer is common to all readers
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
        "type": "universal_dependencies_multi",
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
    "desired_order_of_heads" : ["dependencies"],
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
      "dropout": 0.33,
      "input_dropout_word": 0.33,
      },
    "heads": {
      "dependencies": {
        "type": "singleview_parser",
        "encoder_dim": transformer_dim,
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
    "validation_metric": "+multi_dependencies",
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
