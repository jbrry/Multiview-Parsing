//local transformer_model = std.extVar("MODEL_NAME");
local transformer_model = "bert-base-multilingual-cased";
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

local feedforward_common = {
  "activations": "elu",
  "dropout": 0.33,
  "hidden_dims": 200,
  "input_dim": encoder_dim,
  "num_layers": 1
};

{
  "dataset_reader": {
      "type": "multitask",
      "readers": {
        "ga_idt": reader_common {
          "type": "universal_dependencies_multi",
          "mono_token_indexers": {
            "tokens": {
              "type": "single_id",
              "namespace": "ga_idt_tokens"
            },
            //"token_characters": {
            //  "type": "characters",
            //  "min_padding_length": 5,
            //  "namespace": "ga_idt_token_characters"
            //}
          }
        },
        "en_lines": reader_common {
          "type": "universal_dependencies_multi",
            "mono_token_indexers": {
            "tokens": {
              "type": "single_id",
              "namespace": "en_lines_tokens"
            },
            //"token_characters": {
            //  "type": "characters",
            //  "min_padding_length": 5,
            //  "namespace": "en_lines_token_characters"
            //}
          }
        }
      }
  },

  "train_data_path": {
    "ga_idt": "data/ud-treebanks-v2.5/UD_Irish-IDT/ga_idt-ud-train.conllu",
    "en_lines": "data/ud-treebanks-v2.5/UD_English-LinES/en_lines-ud-train.conllu",
  },
  "validation_data_path": {
    "ga_idt": "data/ud-treebanks-v2.5/UD_Irish-IDT/ga_idt-ud-dev.conllu",
    "en_lines": "data/ud-treebanks-v2.5/UD_English-LinES/en_lines-ud-dev.conllu",
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

      "mono_embedders": {
      
      "ga_idt": {
          "token_embedders" : {
            "tokens": {
              "type": "embedding",
              "vocab_namespace": "ga_idt_tokens",
              "embedding_dim": 25,
            }
          }
      },

      "en_lines": {
        "token_embedders" : {
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "en_lines_tokens",
            "embedding_dim": 25,
          }
        }
      }
    },

      // #=================
      "mono_encoders": {
          "ga_idt": {
            "type": "stacked_bidirectional_lstm",
            "input_size": 25,
            "hidden_size": 100,
            "num_layers": 2,
            "recurrent_dropout_probability": 0.3,
            "use_highway": true
          },

          "en_lines": {
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
            "input_size": 768 + 200,
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
    //"num_gradient_accumulation_steps": gradient_accumulation_steps,
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
