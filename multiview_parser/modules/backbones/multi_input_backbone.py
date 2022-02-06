from typing import Dict, Optional

import torch
from torch.nn.modules import Dropout

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules.backbones.backbone import Backbone
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout

from allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder,
)
from allennlp.nn import util


@Backbone.register("multi_input")
class MultiInputBackbone(Backbone):
    """
    This `Backbone` embeds and encodes input text based on multiple inputs or 'views'.
    For example, one view may comprise of a shared vocabulary and another may be 
    a dataset-specific vocabulary.

    Registered as a `Backbone` with name "multi_input".

    # Parameters

    vocab : `Vocabulary`
        Necessary for converting input ids to strings in `make_output_human_readable`.  If you set
        `output_token_strings` to `False`, or if you never call `make_output_human_readable`, then
        this will not be used and can be safely set to `None`.
    token_embedders : Dict[str, TextFieldEmbedder]
        A dictionary mapping a key of a dataset name to the appropriate TextFieldEmbedder. In this way,
        a dataset will have its own embeddings. NOTE: in order to do this, in your dataset reader, please make
        sure each dataset has its own token_indexer so that it will have different indexing and thus embeddings.
    encoder: Seq2SeqEncoder
        The encoder which encodes some embedded text. At the moment, there is just a single encoder but it is
        possible that we will introduce multiple encoders (one for each dataset).
    output_token_strings : `bool`, optional (default = `True`)
        If `True`, we will add the input token ids to the output dictionary in `forward` (with key
        "token_ids"), and convert them to strings in `make_output_human_readable` (with key
        "tokens").  This is necessary for certain demo functionality, and it adds only a trivial
        amount of computation if you are not using a demo.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        *,
        token_embedders: Dict[str, TextFieldEmbedder],
        encoder: Seq2SeqEncoder,
        output_token_strings: bool = True,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        vocab_namespace: str = "tags",
    ) -> None:
        super().__init__()
        self._vocab = vocab
        self._namespace = vocab_namespace

        self._token_embedders = token_embedders
        self._encoder = encoder
        self._output_token_strings = output_token_strings

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

    def forward(self,
                backbone_arguments,
                ) -> Dict[str, torch.Tensor]:  # type: ignore
        
        if len(backbone_arguments["words_polyglot"]) != 1:
            raise ValueError(
                "Text is only compatible with using a single TokenIndexer"
            )

        if backbone_arguments["dataset"] is None:
            raise ConfigurationError(
                    "Backbone arguments is missing 'dataset' MetadataField; "
                    "Use the universal_dependencies_meta dataset reader.")
        
        dataset_keys = ["polyglot"]
        batch_dataset = backbone_arguments["dataset"][0]
        dataset_keys.append(batch_dataset)

        outputs = {}
        for dataset_key in dataset_keys:
            # Embed the input words with the appropriate token embedder
            words = backbone_arguments[f"words_{dataset_key}"]
            token_embedder = self._token_embedders[f"{dataset_key}_embedder"]
            embedded_text_input = token_embedder(words)
            embedded_text_input = self._input_dropout(embedded_text_input)
            print("Embedded text", embedded_text_input.size())

            # Encode the embedded text
            mask = util.get_text_field_mask(words)
            encoded_text = self._encoder(embedded_text_input, mask)
            encoded_text = self._dropout(encoded_text)
            print("Encoded text", encoded_text.size())
            
            outputs[f"encoded_text_{dataset_key}"] = encoded_text
            outputs["encoded_text_mask"] = mask

        if self._output_token_strings:
            outputs["token_ids"] = util.get_token_ids_from_text_field_tensors(words)

        return outputs

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not self._output_token_strings:
            return output_dict

        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self._vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict