local transformer_model = std.extVar("MODEL_NAME");
//local transformer_model = "bert-base-multilingual-cased";
//local transformer_model = "/home/jbarry/ga_BERT/Irish-BERT/models/ga_bert/output/pytorch/gabert/conll17_gdrive_NCI_oscar_paracrawl_filtering_basic+char-1.0+lang-0.8";
local max_length = 128;
local transformer_dim = 768;

local tokenizer_kwargs = {
  "tokenizer_kwargs": {
    "do_lower_case": false,
    "tokenize_chinese_chars": true,
    "strip_accents": false,
    "clean_text": true,
  }
};

{
    //"random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
    //"numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
    //"pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),  
    "dataset_reader":{
        "type":"universal_dependencies",
        "token_indexers": {
        "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": transformer_model,
          "max_length": max_length,
          "tokenizer_kwargs": tokenizer_kwargs
        }
        }
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("DEV_DATA_PATH"),
    "model": {
      "type": "biaffine_parser",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": transformer_model,
            "max_length": max_length
          }
        }
      },
      "encoder": {
        "type": "pass_through",
        "input_dim": 768
      },
      "use_mst_decoding_for_validation": true,
      "arc_representation_dim": 500,
      "tag_representation_dim": 100,
      "dropout": 0.33,
      "input_dropout": 0.33,
      "initializer": {
        "regexes": [
          [".*projection.*weight", {"type": "xavier_uniform"}],
          [".*projection.*bias", {"type": "zero"}],
          [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
          [".*tag_bilinear.*bias", {"type": "zero"}],
          [".*weight_ih.*", {"type": "xavier_uniform"}],
          [".*weight_hh.*", {"type": "orthogonal"}],
          [".*bias_ih.*", {"type": "zero"}],
          [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
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
      "patience": 10,
      "validation_metric": "+LAS",
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
    }
}