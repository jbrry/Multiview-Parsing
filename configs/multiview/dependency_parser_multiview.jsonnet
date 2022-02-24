local num_gpus = 1;
local gpu_batch_size = 8;
local effective_batch_size = gpu_batch_size * num_gpus;
local num_epochs = 50;
local patience = 8;
local num_gradient_accumulation_steps = 32 / effective_batch_size;
local max_length = 128;
local transformer_dim = 768;
local encoder_dim = transformer_dim;

// the pretrained transformer is common to all readers
local reader_common = {
  "token_indexers": {
    "tokens": {
      "type": "pretrained_transformer_mismatched",
      "model_name": "",
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
      },     
    }
  },
  "train_data_path": {
      "TBID_PLACEHOLDER": "",
  },
  "validation_data_path": {
      "TBID_PLACEHOLDER": "",
  },
  "test_data_path": {
      "TBID_PLACEHOLDER": "",
  },

  "model": {
    "type": "multitask_v2",
    "multiple_heads_one_data_source": true,
    "desired_order_of_heads" : [],
    "allowed_arguments": {
        "backbone": ["words"],
        "TBID_PLACEHOLDER": ["encoded_text", "task", "mask", "upos", "metadata", "head_tags", "head_indices"],
        "multi_dependencies": ["encoded_text", "task", "mask", "upos", "metadata", "head_tags", "head_indices"],
        "meta_dependencies": ["other_module_inputs", "task", "mask", "upos", "metadata", "head_tags", "head_indices"]
    },

    "backbone": {
      "type": "transformer",
        "text_field_embedder": {
          "token_embedders": {
            "tokens": {
              "type": "pretrained_transformer_mismatched",
              "model_name": "",
              "max_length": max_length
            }
          }
        },
      "dropout": 0.33,
      "input_dropout_word": 0.33,
    },
    "heads": {
      "TBID_PLACEHOLDER": {
        "type": "multiview_parser",
        "encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": transformer_dim,
          "hidden_size": 400,
          "num_layers": 2,
          "recurrent_dropout_probability": 0.33,
          "use_highway": true
        },
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33
      },
      "multi_dependencies": {
        "type": "multiview_parser",
        "encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": transformer_dim,
          "hidden_size": 400,
          "num_layers": 2,
          "recurrent_dropout_probability": 0.33,
          "use_highway": true
        },
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33
      },
      "meta_dependencies": {
        "type": "multiview_meta_parser",
        "meta_encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": 800 + 800,
          "hidden_size": 400,
          "num_layers": 2,
          "recurrent_dropout_probability": 0.33,
          "use_highway": true
        },
        "tag_representation_dim": 100,
        "arc_representation_dim": 500,
        "use_mst_decoding_for_validation": true,
        "dropout": 0.33,
        "first_encoded_text_source": "dependencies_module_text",
        "second_encoded_text_source": "multi_dependencies_module_text",
	      "use_cross_stitch": false
      }
    }
  },
  "data_loader": {
    "type": "multitask",
    "scheduler": {
      "batch_size": gpu_batch_size
    },
    "shuffle": true,
  },
  "trainer": {
    "num_epochs": num_epochs,
    "grad_norm": 5.0,
    "patience": patience,
    "cuda_device": 0,
    "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
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
  "random_seed": "",
  "numpy_seed": "",
  "pytorch_seed": "",
}
