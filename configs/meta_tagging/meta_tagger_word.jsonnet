// Meta tagger with word features
{
    "dataset_reader":{
        "type":"universal_dependencies_word_char",
        "word_token_indexer" : {
          "tokens" : {
            "type": "single_id",
            "namespace": "token_vocab"
          }
        }     
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("DEV_DATA_PATH"),
    //train_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-train.conllu',
    //validation_data_path: 'data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-dev.conllu',
    
    "model": {
      "type": "multitask_word_char",
      "allowed_arguments": {
        "backbone" : ["words"],
        "word": ["encoded_words", "mask", "pos_tags", "metadata"]
    },

    "backbone": {
      "type" : "word_char",
        "word_embedder": {
          "token_embedders": {          
          "tokens": {
            "type": "embedding",
            "vocab_namespace": "token_vocab",
            "embedding_dim": 100,
          }
          }
        },
        "word_encoder": {
          "type": "stacked_bidirectional_lstm",
          "input_size": 100,
          "hidden_size": 200,
          "num_layers": 2,
          "recurrent_dropout_probability": 0.33,
          "use_highway": true
        },
        "dropout": 0.33,
        "input_dropout_word": 0.33
    },
    "heads": {
      "word": {
        "type": "word_tagger",
            "feedforward": {
                "activations": "elu",
                "dropout": 0.33,
                "hidden_dims": 200,
                "input_dim": 400,
                "num_layers": 1
            },
        "encoder_dim": 400,
        "dropout": 0.33
      }
    }
    },
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : 32
      }
    },
    "trainer": {
      "cuda_device": 0,
      "num_epochs": 75,
      "grad_norm": 5.0,
      "patience": 6,
      "validation_metric": "+word_accuracy",
      "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "checkpointer": {
        "num_serialized_models_to_keep": 1,
    },
    }
}
