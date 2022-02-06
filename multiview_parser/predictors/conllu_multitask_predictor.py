"""
based on the implementations in:

https://github.com/Hyperparticle/udify/blob/master/udify/predictors/predictor.py
https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/predictors/biaffine_dependency_parser.py

"""

from typing import Dict, Any, List, Tuple


from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


@Predictor.register("conllu-multitask-predictor")
class ConlluMultitaskPredictor(Predictor):
    """
    Predictor for CoNLL-U files.
    """

    def __init__(
        self,
        model: Model,
        dataset_reader: DatasetReader,
        language: str = "en_core_web_sm",
        head_name: str = "dependencies",
        write_metadata: bool = True,
    ) -> None:
        super().__init__(model, dataset_reader)

        self.head_name = head_name
        self.write_metadata = write_metadata

        self.CONLLU_FIELDS = [f"{self.head_name}_ids", f"{self.head_name}_words", f"{self.head_name}_lemmas", "upos_upos", "xpos_xpos", "feats_feats",
                    f"{self.head_name}_predicted_heads", f"{self.head_name}_predicted_dependencies", f"{self.head_name}_deps", f"{self.head_name}_misc"]

        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)
    
    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a dependency parse for the given sentence.
        # Parameters
        sentence The sentence to parse.
        # Returns
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self,
                        json_dict: JsonDict,
                        ) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"tokens"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = sentence.split()
        tokens = str(tokens)
        return self._dataset_reader.text_to_instance(tokens)

    def predict_json(self, instances: List[JsonDict]) -> List[JsonDict]:
        raise NotImplementedError

    def predict_batch_json(self, instances: List[JsonDict]) -> List[JsonDict]:
        raise NotImplementedError

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        if "@@UNKNOWN@@" not in self._model.vocab._token_to_index[f"{self.head_name}_head_tags"]:
            # Handle cases where the labels are present in the test set but not training set
            for instance in instances:
                self._predict_unknown(instance)
        
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    def predict_instance(self, instance: Instance) -> JsonDict:
        if "@@UNKNOWN@@" not in self._model.vocab._token_to_index[f"{self.head_name}_head_tags"]:
            # Handle cases where the labels are present in the test set but not training set
            # https://github.com/Hyperparticle/udify/blob/b6a1173e7e5fc1e4c63f4a7cf1563b469268a3b8/udify/predictors/predictor.py
            self._predict_unknown(instance)

        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def _predict_unknown(self, instance: Instance):
        """
        Maps each unknown label in each namespace to a default token
        :param instance: the instance containing a list of labels for each namespace
        from: https://github.com/Hyperparticle/udify/blob/b6a1173e7e5fc1e4c63f4a7cf1563b469268a3b8/udify/predictors/predictor.py
        """
        def replace_tokens(instance: Instance, namespace: str, token: str):
            if namespace not in instance.fields:
                return

            instance.fields[namespace].labels = [label
                                                 if label in self._model.vocab._token_to_index[namespace]
                                                 else token
                                                 for label in instance.fields[namespace].labels]

        replace_tokens(instance, "lemmas", "_")
        replace_tokens(instance, "feats", "_")
        replace_tokens(instance, "xpos", "_")
        replace_tokens(instance, "upos", "NOUN")
        replace_tokens(instance, "head_tags", "case")


    def dump_line(self, outputs: JsonDict) -> str:
        word_count = len([word for word in outputs[f"{self.head_name}_words"]])

        lines = zip(*[outputs[k] if k in outputs else ["_"] * word_count
                      for k in self.CONLLU_FIELDS])

        multiword_map = None
        if outputs[f"{self.head_name}_multiword_ids"] is not "None":
            multiword_ids = [[id] + [int(x) for x in id.split("-")] for id in outputs[f"{self.head_name}_multiword_ids"]]
            multiword_forms = outputs[f"{self.head_name}_multiword_forms"]
            multiword_map = {start: (id_, form) for (id_, start, end), form in zip(multiword_ids, multiword_forms)}

        output_lines = []
        for i, line in enumerate(lines):
            line = [str(l) for l in line]

            # Handle multiword tokens
            if multiword_map and i+1 in multiword_map:
               id_, form = multiword_map[i+1]
               row = f"{id_}\t{form}" + "".join(["\t_"] * 8)
               output_lines.append(row)

            row = "\t".join(line)
            output_lines.append(row)
        
        if self.write_metadata:
            conllu_metadata = outputs[f"{self.head_name}_conllu_metadata"]
            output_lines = conllu_metadata + output_lines

        return "\n".join(output_lines) + "\n\n"
