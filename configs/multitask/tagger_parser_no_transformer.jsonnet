local encoder_dim = 400;

local batch_size = 16;
local gradient_accumulation_steps = 32;

local reader_common = {
    "token_indexers": {
        "tokens": {
            "type": "single_id",
        },
        "token_characters": {
            "type": "characters",
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
          "type": "universal_dependencies_all_features",
        }
      }
  },

  "train_data_path": {
    "ga_idt": "data/ud-treebanks-v2.8/UD_Irish-IDT/ga_idt-ud-train.conllu",
  },
  "validation_data_path": {
    "ga_idt": "data/ud-treebanks-v2.8/UD_Irish-IDT/ga_idt-ud-dev.conllu",
  },
  "test_data_path": {
    "ga_idt": "data/ud-treebanks-v2.8/UD_Irish-IDT/ga_idt-ud-test.conllu",
  },

  "model": {
    "type": "multitask_v2",
    "multiple_heads_one_data_source": true,
    "desired_order_of_heads" : ["upos", "xpos", "feats", "dependencies"],
    "backbone": {
      "type": "rnn",
        "word_embedder": {
          "token_embedders": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 100,
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": 32
                },
                "encoder": {
                "type": "lstm",
                "input_size": 32,
                "hidden_size": 64,
                "num_layers": 3,
                "bidirectional": true
                }
              }
          },
        },

    "word_encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 100 + 128,
      "hidden_size": 200,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.33,
      "use_highway": true
    },
    "dropout": 0.33,
    "input_dropout_word": 0.33,
    "input_dropout_character": 0.05
    },
    "heads": {
      "upos": {
        "type": "multitask_tagger",
        "task": "upos",
        "encoder_dim": encoder_dim,
        "feedforward": feedforward_common
      },
      "xpos": {
        "type": "multitask_tagger",
        "task": "xpos",
        "encoder_dim": encoder_dim,
        "feedforward": feedforward_common
      },
      "feats": {
        "type": "multitask_tagger",
        "task": "feats",
        "encoder_dim": encoder_dim,
        "feedforward": feedforward_common
      },
      "dependencies": {
        "type": "multitask_parser",
        "encoder_dim": encoder_dim + 200 + 200 + 200,
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
    "validation_metric": "+dependencies_LAS",
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
