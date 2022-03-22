# MPM-Parser
Mixed Polyglot Monolingual Parsing

### Installation

To work with the main version of AllenNLP:

Using `venv`

```
mkdir mvenv
python3 -m venv ./mvenv/
source mvenv/bin/activate

pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install conllu
```


```
conda create -n allennlp python=3.7
conda activate allennlp
pip install allennlp
pip install allennlp-models
```

If you need to modify any of the existing AllenNLP code, you can install AllenNLP from source:
```
conda create -n meta_parser python=3.7
conda activate meta_parser

git clone https://github.com/allenai/allennlp.git
cd allennlp
pip install --editable .
pip install -r dev-requirements.txt
git checkout vision
```
Install `allennlp-models`
```
git clone https://github.com/allenai/allennlp-models.git
cd allennlp-models
ALLENNLP_VERSION_OVERRIDE='allennlp' pip install -e .
pip install -r dev-requirements.txt
```

### Obtain Universal Dependencies data
The following will download the UD v2.8 data:
```
./scripts/download_ud_data.sh
```

### Usage

Once you have the necessary software installed and the UD data downloaded to the `data` folder, you should then be able to train a model.

#### Train a dependency parser
Run `./scripts/train_dependency_parser.sh` with the following arguments:
- `tbid(s)` the treebank ids to use, can be a single tbid, e.g. `ga_idt` or a number of colon-separated tbids `ga_idt:da_ddt:en_lines`.
- `model_type`: the model type to use, this will select the appropriate parent folder in the `configs` directory, to see what model types are available, see the sub directory names in `configs`, examples include: `dependency_parser`, `meta_parser`, `multitask`.
- `feature_type` the types of feature being used, this information is in some places redundant, but if you are using a transformer, you can just say `transformer`. For certain applications, this argument is used to find the appropriate configuration file, but this isn't consistent everywhere.
- `model_name`, the name of the pre-trained model to use on HuggingFace models https://huggingface.co/models
- `metadata`, some optional metadata to include for this training run. Perhaps you want to say, that the model is trained on all of the Irish data, so you could put the string "all-ga" to help identify this run. This can be any string to help identify this run.

An example command would be:

```
./scripts/train_dependency_parser.sh ga_idt multitask transformer DCU-NLP/bert-base-irish-cased-v1 all-ga
```
