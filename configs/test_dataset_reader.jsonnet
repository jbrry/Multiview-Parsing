{
  "random_seed": 711,
  
  dataset_reader: {
    "type": 'universal_dependencies_source_target',
      "tbids": ["en_ewt", "en_lines"],
      "alternate": true,
      "instances_per_file": 32,
      "is_first_pass_for_vocab": true,    
      "lazy": false,
  },
  
  train_data_path: 'data/ud-treebanks-v2.6/**/*-ud-train.conllu',
  validation_data_path: 'data/ud-treebanks-v2.6/**/*-ud-dev.conllu',
  test_data_path: 'data/ud-treebanks-v2.6/**/*-ud-test.conllu',
  
  model: {
    type: 'dummy_lstm'
  },
  
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 128      
    }
  },
  
  trainer: {
    cuda_device: 0
  }
}
