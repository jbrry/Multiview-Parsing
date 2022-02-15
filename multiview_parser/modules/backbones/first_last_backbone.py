from typing import Dict, Optional, List

import torch
from torch.nn.modules import Dropout

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules.backbones.backbone import Backbone
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder 
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout

from allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder,
)
from allennlp.nn import util

@Backbone.register("first_last")
class FirstLastBackbone(Backbone):
    """

    Registered as a `Backbone` with name "first_last".
    A TextFieldEmbedder is used to encode the word representations, where the 
    TextFieldEmbedder should be a pretrained_transformer*. In this way, we do not have a second encoder
    for the word features. There are, however, encoders to encode the character representations which are 
    then concatenated with the output of a pretrained transformer.

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
        first_last_embedder: TextFieldEmbedder = None,
        output_token_strings: bool = False,
        dropout: float = 0,
        input_dropout_word: float = 0,
    ) -> None:
        super().__init__()
        
        self._vocab = vocab
        # source embedders/encoder
        self._first_last_embedder = first_last_embedder
        self._output_token_strings = output_token_strings
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout_word = Dropout(input_dropout_word)

    def forward(self,
                metadata: List[Dict] = None,
                task: List[str] = None,
                words: TextFieldTensors = None,
                ) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        words: tensor of words.
        token_characters: tensor of characters which comprise the words of the sentence.
        sentence_characters: tensor of characters spanning the whole sentence.

        Returns:
            encoded_text: The above features will be encoded and concatenated together.
        """

        batch_tbid = task[0]

        if words and len(words) != 1:
            raise ValueError(
                "Pretrained Transformer is only compatible with using single TokenIndexers"
            )

        mask = util.get_text_field_mask(words)

        if self._first_last_embedder:
            embedded_text_input = self._first_last_embedder(words)
            # NOTE: this assumes 2 concatenated input sequences each of the same length
            split_size = int(len(embedded_text_input) / 2)
            first_last_embedded_text_input = torch.split(embedded_text_input, split_size)
            first_embedded_text_input = first_last_embedded_text_input[0]
            last_embedded_text_input = first_last_embedded_text_input[1]

            first_embedded_text_input = self._input_dropout_word(first_embedded_text_input)
            last_embedded_text_input = self._input_dropout_word(last_embedded_text_input)

        # Pack the encoded text representations
        encoded_text = {}
        encoded_text["first_encoded_text"] = first_embedded_text_input
        encoded_text["last_encoded_text"] = last_embedded_text_input

        outputs = {
            "encoded_text": encoded_text,
            "mask": mask
        }

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
