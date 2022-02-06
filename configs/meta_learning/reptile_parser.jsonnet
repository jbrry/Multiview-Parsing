local reader_common = {
    "token_indexers": {
        "tokens": {
            "type": "single_id",
        }
    }
};

{
    "dataset_reader": {
        "type": "multitask",
        "readers": {
            "en_lines": reader_common {
                "type": "universal_dependencies_all_features",
            },
            "fi_ftb": reader_common {
                "type": "universal_dependencies_all_features",
            }
        }
    },
    // Sample a language l, from L.
    // for that language (task) sample n sentences
    // and put them into a batch of a predefined size.
    // Do a forward pass on that batch.
    // Repeat for the other sampled languages.
    "data_loader": {
        "type": "multitask",
        "scheduler": {
            "type": "homogeneous_roundrobin",
            "batch_size": {"en_lines": 32, "fi_ftb": 32},
    },
    "sampler": "uniform",
    },
    "train_data_path": {
        "en_lines": "data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-train.conllu",
        "fi_ftb": "data/ud-treebanks-v2.6/UD_Finnish-FTB/fi_ftb-ud-train.conllu",
    },
    "validation_data_path": {
        "en_lines": "data/ud-treebanks-v2.6/UD_English-LinES/en_lines-ud-dev.conllu",
        "fi_ftb": "data/ud-treebanks-v2.6/UD_Finnish-FTB/fi_ftb-ud-dev.conllu",
    },
    "model": {
        "type": "pos_tagger",
        "dropout": 0.33,
        "task": "upos",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 100,
                }
       },
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 100,
      "hidden_size": 200,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.33,
      "use_highway": true
    },
    "feedforward": {
        "activations": "elu",
        "dropout": 0.33,
        "hidden_dims": 200,
        "input_dim": 400,
        "num_layers": 1
    }
    },
    "trainer": {
        "type": "metatrainer",
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "checkpointer": {
            "num_serialized_models_to_keep": 1,
        },
        "validation_metric": "+accuracy",
        "num_epochs": 50,
        "grad_norm": 5.0,
        "patience": 25,
    }
}