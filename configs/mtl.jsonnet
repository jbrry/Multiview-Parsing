{
  "dataset_reader": {
        "type": "universal_dependencies_meta",
        "dataset_ids": ["en_ewt", "en_lines"],
        "use_polyglot": true,
        "alternate": false,
        "instances_per_file": 32,
        "is_first_pass_for_vocab": true,
        // token_indexers which wraps multiple dataset-specifc token_indexers
        "token_indexers" : {
          "polyglot_token_indexers": {
            "tokens": {
              "type": "single_id",
              "namespace": "polyglot_tokens"
            }
          },          
          "en_ewt_token_indexers": {
            "tokens": {
              "type": "single_id",
              "namespace": "en_ewt_tokens"
            }
          },
          "en_lines_token_indexers": {
            "tokens": {
              "type": "single_id",
              "namespace": "en_lines_tokens"
            }
          }
      }
  },
  train_data_path: 'tests/data/ud-treebanks-v2.6/**/*-ud-train.conllu',
  validation_data_path: 'tests/data/ud-treebanks-v2.6/**/*-ud-dev.conllu',

  "model": {
    "type": "multi_task_meta_model",
    "allowed_arguments": {
      // we are just concerned with embedding / encoding words in the backbone
      "backbone" : ["words_polyglot", "words_en_ewt", "words_en_lines", "dataset"],
      "polyglot": ["encoder", "encoded_text_polyglot", "encoded_text_mask", "pos_tags_polyglot", "head_indices_polyglot", "head_tags_polyglot", "metadata"]
    },
    "backbone": {
    "type" : "multi_input",
      // wrap token embedders in a dict 
      "token_embedders": {
      "polyglot_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "polyglot_tokens",
            "embedding_dim": 25,
            "trainable": true
          }
        }
      },
      "en_ewt_embedder": {
        "token_embedders" : {
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "en_ewt_tokens",
            "embedding_dim": 25,
            "trainable": true
          }
        }
      },
      "en_lines_embedder": {
        "token_embedders" : {
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "en_lines_tokens",
            "embedding_dim": 25,
            "trainable": true
          }
        }
      }
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 25,
      "hidden_size": 200,
      "num_layers": 3,
      "recurrent_dropout_probability": 0.3,
      "use_highway": true
    }
    },
    "heads": {
      "polyglot": {
        "type": "polyglot_parser",
        "encoder_dim": 400,
        "tag_representation_dim": 128,
        "arc_representation_dim": 128,
      }//,
      //"monolingual": {
      //  "type": "monolingual_parser",
      //  "tag_representation_dim": 128,
      //  "arc_representation_dim": 128,
      //},
      //"meta": {
      //  "type": "meta_parser",
      //  "tag_representation_dim": 128,
      //  "arc_representation_dim": 128,
      //},
    }
},
"data_loader": {
    "batch_sampler": {
    "type": "homogeneous_bucket",
    "partition_key": "dataset",
    "batch_size" : 32
    }
},
    "trainer": {
      "num_epochs": 200,
      "grad_norm": 5.0,
      "patience": 200,
      "cuda_device": -1,
      "validation_metric": "+polyglot_LAS",
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.9]
      }
    }
}