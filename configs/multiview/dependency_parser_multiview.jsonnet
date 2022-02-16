local transformer_model = std.extVar("MODEL_NAME");
local max_length = 128;
local transformer_dim = 768;
local encoder_dim = transformer_dim;

local batch_size = 16;

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
      // duplicate
      "nl_alpino": reader_common {
        "type": "universal_dependencies_all_features",
      },
      // duplicate
      "af_afribooms": reader_common {
        "type": "universal_dependencies_all_features",
      },
      // duplicate
      "nl_lassysmall": reader_common {
        "type": "universal_dependencies_all_features",
      },
      // duplicate
      "de_gsd": reader_common {
        "type": "universal_dependencies_all_features",
      },    
    }
  },
    "train_data_path": {
        "nl_alpino": "data/ud-treebanks-v2.8/UD_Dutch-Alpino/nl_alpino-ud-train.conllu",
        "af_afribooms": "data/ud-treebanks-v2.8/UD_Afrikaans-AfriBooms/af_afribooms-ud-train.conllu",
        "nl_lassysmall": "data/ud-treebanks-v2.8/UD_Dutch-LassySmall/nl_lassysmall-ud-train.conllu",
        "de_gsd": "data/ud-treebanks-v2.8/UD_German-GSD/de_gsd-ud-train.conllu",
    },
    "validation_data_path": {
        "nl_alpino": "data/ud-treebanks-v2.8/UD_Dutch-Alpino/nl_alpino-ud-dev.conllu",
        "af_afribooms": "data/ud-treebanks-v2.8/UD_Afrikaans-AfriBooms/af_afribooms-ud-dev.conllu",
        "nl_lassysmall": "data/ud-treebanks-v2.8/UD_Dutch-LassySmall/nl_lassysmall-ud-dev.conllu",
        "de_gsd": "data/ud-treebanks-v2.8/UD_German-GSD/de_gsd-ud-dev.conllu",
    },
    "test_data_path": {
        "nl_alpino": "data/ud-treebanks-v2.8/UD_Dutch-Alpino/nl_alpino-ud-test.conllu",
        "af_afribooms": "data/ud-treebanks-v2.8/UD_Afrikaans-AfriBooms/af_afribooms-ud-test.conllu",
        "nl_lassysmall": "data/ud-treebanks-v2.8/UD_Dutch-LassySmall/nl_lassysmall-ud-test.conllu",
        "de_gsd": "data/ud-treebanks-v2.8/UD_German-GSD/de_gsd-ud-test.conllu",
    },

  "model": {
    "type": "multitask_v2",
    "multiple_heads_one_data_source": true,
      "allowed_arguments": {
        "backbone": ["words"],
        "nl_alpino_dependencies": ["encoded_text", "task", "mask", "upos", "metadata", "head_tags", "head_indices"],
        "af_afribooms_dependencies": ["encoded_text", "task", "mask", "upos", "metadata", "head_tags", "head_indices"],
        "nl_lassysmall_dependencies": ["encoded_text", "task", "mask", "upos", "metadata", "head_tags", "head_indices"],
        "de_gsd_dependencies": ["encoded_text", "task", "mask", "upos", "metadata", "head_tags", "head_indices"],
        "multi_dependencies": ["encoded_text", "task", "mask", "upos", "metadata", "head_tags", "head_indices"],
        "meta_dependencies": ["other_module_inputs", "task", "mask", "upos", "metadata", "head_tags", "head_indices"]
    },

    //"desired_order_of_heads" : ["singleview_dependencies"],
    "backbone": {
      "type": "transformer",
        "text_field_embedder": {
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
      // this block needs to be duplicated
      "nl_alpino_dependencies": {
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
      // this block needs to be duplicated
      "af_afribooms_dependencies": {
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
      // this block needs to be duplicated
      "nl_lassysmall_dependencies": {
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
      // this block needs to be duplicated
      "de_gsd_dependencies": {
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
