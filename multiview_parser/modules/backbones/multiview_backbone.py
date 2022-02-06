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

@Backbone.register("multiview")
class MultiviewBackbone(Backbone):
    """

    Registered as a `Backbone` with name "multiview".
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
        multi_embedder: TextFieldEmbedder = None,
        multi_encoder: Seq2SeqEncoder = None,
        text_field_embedder: TextFieldEmbedder = None,
        mono_embedders: Dict[str, TextFieldEmbedder] = None,
        mono_encoders: Dict[str, Seq2SeqEncoder] = None,
        mono_token_character_embedders: Dict[str, TextFieldEmbedder] = None,
        mono_sentence_character_embedders: Dict[str, TextFieldEmbedder] = None,
        mono_token_character_encoders: Dict[str, Seq2VecEncoder] = None,
        mono_sentence_character_encoders: Dict[str, Seq2SeqEncoder] = None,
        output_token_strings: bool = False,
        dropout: float = 0.0,
        input_dropout_word: float = 0.0,
        input_dropout_character: float = 0.0,
    ) -> None:
        super().__init__()
        
        self._vocab = vocab
        self._padding_index = 0 # NOTE: some indexers pad things as -1

        # source embedders/encoder
        self._multi_embedder = multi_embedder
        self._multi_encoder = multi_encoder

        # mono token embedder
        if mono_embedders != None:
            self._mono_embedders = torch.nn.ModuleDict({
                tbid: embedder
                for tbid, embedder in mono_embedders.items()
            })
        else:
            self._mono_embedders = mono_embedders

        # mono token-char embedder
        if mono_token_character_embedders != None:
            self._mono_token_character_embedders = torch.nn.ModuleDict({
                tbid: embedder
                for tbid, embedder in mono_token_character_embedders.items()
            })
        else:
            self._mono_token_character_embedders = mono_token_character_embedders

        # mono sentence-char embedder
        if mono_sentence_character_embedders != None:
            self._mono_sentence_character_embedders = torch.nn.ModuleDict({
                tbid: embedder
                for tbid, embedder in mono_sentence_character_embedders.items()
            })
        else:
            self._mono_sentence_character_embedders = mono_sentence_character_embedders

        # mono encoders
        if mono_encoders != None:
            self._mono_encoders = torch.nn.ModuleDict({
                tbid: encoder
                for tbid, encoder in mono_encoders.items()
            })
        else:
            self._mono_encoders = mono_encoders

        # token-char encoders
        if mono_token_character_encoders != None:
            self._mono_token_character_encoders = torch.nn.ModuleDict({
                tbid: encoder
                for tbid, encoder in mono_token_character_encoders.items()
            })
        else:
            self._mono_token_character_encoders = mono_token_character_encoders

        # sentence-char encoders
        if mono_sentence_character_encoders != None:
            self._mono_sentence_character_encoders = torch.nn.ModuleDict({
                tbid: encoder
                for tbid, encoder in mono_sentence_character_encoders.items()
            })
        else:
            self._mono_sentence_character_encoders = mono_sentence_character_encoders

        if self._mono_sentence_character_encoders:
            # add a dummy_tensor which we will use for padding, after we have collected the start and end indices of the words.
            for k, encoder in self._mono_sentence_character_encoders.items():
                encoder_dim = encoder.get_output_dim() * 2
                # just do this once, NOTE: this assumes all dims are the same for each task
                break
            self._dummy_tensor = torch.nn.Parameter(torch.zeros([encoder_dim]))

        self._output_token_strings = output_token_strings
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout_word = Dropout(input_dropout_word)
        self._input_dropout_character = Dropout(input_dropout_character)

    def find_word_start_and_end_indices(self, character_tensor, batch_tbid, metadata):
        """
        Iterate over the characters of each sentence to find
        the locations of the start and end characters of each word.
        The start and end locations of each word are then assigned to a tuple.

        character_tensor : TextFieldTensors = Dict[str, Dict[str, torch.Tensor]]
        """

        padding_index = 0
        start_locations = []
        end_locations = []

        character_vocab = self._vocab.get_token_to_index_vocabulary(f"{batch_tbid}_sentence_characters")
        space_index = character_vocab.get(" ")

        # Access the tensor object which is wrapped by two dictionaries.	
        character_tensor = character_tensor["token_characters"]["tokens"]

        # extract sentences from metadata
        sents = []
        for entry in metadata:
            sents.append(entry["words"])

        for sent_idx, sentence in enumerate(character_tensor):
            # raw sent
            sent = sents[sent_idx]
            raw_sent = list(" ".join(sent))

            ci = 0
            # some tokens have inter-word spaces, e.g. "50 000", add them to a list to ignore
            inter_word_spaces = []
            for w in sent:
                for c in w:
                    if c == " ":
                        assert raw_sent[ci] == " ", "trying to ignore a non-space char"
                        inter_word_spaces.append(ci)
                    ci += 1
                # count spaces between words
                ci += 1

            start_of_sentence = True
            have_seen_space_char = False
            reached_eos = False
            mid_sentence = False
            
            sentence_starts = []
            sentence_ends = []
            
            for i, char in enumerate(sentence):
                if i in inter_word_spaces:
                    mid_sentence = True
                    continue
                
                # append first character index
                if start_of_sentence or last_was_space_char:
                    sentence_starts.append(i)
                    start_of_sentence = False
                    last_was_space_char = False
                else:
                    # search for space index
                    if char == space_index:
                        # end is one char before space
                        end_of_word_idx = i - 1
                        sentence_ends.append(end_of_word_idx)
                        last_was_space_char = True
                        have_seen_space_char = True
                        # we found a valid space so we are not currently mid-sentence
                        mid_sentence = False
                    
                    # reached the end of the sentence without an intervening space
                    elif char == self._padding_index:
                        end_of_word_idx = i - 1
                        sentence_ends.append(end_of_word_idx)
                        reached_eos = True
                        # if we've reached the EOS, we should have the same starts and ends
                        assert len(sentence_starts) == len(sentence_ends), \
                        "Reached end of sentence but there are missing starts/ends!"
                        break

            # 1) Sentence may not contain any space characters: append last character of word.
            # if we've reach EOS, this should already be appended
            if not have_seen_space_char:
                # continuous chars take up the full sentence
                if not reached_eos:
                    # we have the start but no end
                    if len(sentence_starts) > len(sentence_ends):
                        sentence_ends.append(i)

            # 2) Some words are unfinished and we can't always rely on padding to infer the end of the sentence
            # or the last character is a singleton: append last character in sentence.
            if len(sentence_starts) > len(sentence_ends):
                if not mid_sentence:
                    sentence_ends.append(i)
            
            if len(sentence_starts) < len(sentence_ends):
                raise ValueError("More ends than starts, something went wrong")
           
            # check we have a start and end for each token
            assert len(sentence_starts) == len(sentence_ends) == len(sent)
            start_locations.append(sentence_starts)
            end_locations.append(sentence_ends)

        assert len(start_locations) == len(end_locations), "difference in starts and ends"

        # Gather the start and end indices into start and end tuples.
        location_tuples = []
        for sentence_starts, sentence_ends in zip(start_locations, end_locations):
            current_sent = []
            for word_start, word_end in zip(sentence_starts, sentence_ends):
                word_start_and_end = (word_start, word_end)
                current_sent.append(word_start_and_end)
            location_tuples.append(current_sent)
        
        assert len(location_tuples) == character_tensor.size(0)
        #print(f"collected {len(location_tuples)} location tuples, for a tensor with {character_tensor.size(0)} elements")
        #print("location tuples", location_tuples, len(location_tuples))
        return location_tuples


    def concatenate_word_start_and_end_indices(self, encoded_characters, word_starts_and_ends):
        """
        Takes in the encoded text of characters and concatenates the specific locations of the start and end of each word.

        encoded_characters: `torch.Tensor`
            A tensor of shape (batch_size, sequence_len, char_hidden_size * 2)
        word_starts_and_ends: `List[List[Tuple[int, int]]]`,
            Contains a list of tuples for each element in the batch, where each tuple contains the
            locations of the first and last character of a word in the sentence, e.g.:
                [(0, 4), (6, 8), (10, 10), (12, 13), (15, 21), (23, 23)]
        """
        # List to store the concatenated first and last characters of each word for each sentence in the batch.
        concatenated_tensor_list = []
        for i, start_end_tuple in enumerate(word_starts_and_ends):
            # store representations for the specific sentence.
            sentence_tensor_list = []
            for start, end in start_end_tuple:
                # word_representation is the first and last character representation of a word concatenated together.
                word_representation = torch.cat((encoded_characters[i][start], encoded_characters[i][end]), 0)
                sentence_tensor_list.append(word_representation)
            concatenated_tensor_list.append(sentence_tensor_list)

        return concatenated_tensor_list


    def pad_character_representations(self, tensors):
        """Because we extract the start and end indices from already-encoded text (which is padded),
        we need to add padding to our condensed representation which are just the start and end locations concatenated together."""

        lengths = [len(tensor) for tensor in tensors]
        max_len = max(lengths)

        padded = []
        for sentence in tensors:
            padding = []
            if len(sentence) < max_len:
                diff = max_len - len(sentence)
                for i in range(diff):
                    padding.append(self._dummy_tensor)
                padded_sentence = sentence + padding
                padded.append(padded_sentence)
            else:
                # No padding required for sentences which are of length max_len
                padded.append(sentence)

        return padded


    def convert_to_stacked_tensor(self, padded):
        """Converts a list-of-lists of tensors to a single stacked tensor."""
        
        stack = []
        for padded_tensor_list in padded:
            # converts a list to a tensor object
            tensor = torch.stack(padded_tensor_list)
            # append tensor to stack
            stack.append(tensor)

        # stack all tensors
        stacked = torch.stack(stack)
        encoded_first_and_last_characters = stacked

        return encoded_first_and_last_characters


    def forward(self,
                metadata: List[Dict] = None,
                task: List[str] = None,
                words: TextFieldTensors = None,
                multi_words: TextFieldTensors = None,
                token_characters: TextFieldTensors = None,
                sentence_characters: TextFieldTensors = None,
                tc: TextFieldTensors = None,
                ) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        words: tensor of words.
        token_characters: tensor of characters which comprise the words of the sentence.
        sentence_characters: tensor of characters spanning the whole sentence.

        Returns:
            encoded_text: The above features will be encoded and concatenated together.
        """

        batch_tbid = task[0]

        if multi_words and len(multi_words) != 1:
            raise ValueError(
                "Pretrained Transformer is only compatible with using single TokenIndexers"
            )

        if token_characters and len(token_characters) != 1:
            raise ValueError(
                "WordCharBackbone is only compatible with using single TokenIndexers"
            )

        mask = util.get_text_field_mask(words)

        # Multi view
        if self._multi_embedder:
            multi_embedded_text_input = self._multi_embedder(multi_words)
            multi_embedded_text_input = self._input_dropout_word(multi_embedded_text_input)

            # pretrained transformers don't require a second encoder but we keep this here in case you want to encode its output.
            if self._multi_encoder:
                multi_encoded_text = self._multi_encoder(multi_embedded_text_input, mask)
                multi_encoded_text = self._dropout(multi_encoded_text)
            else:
                multi_encoded_text = multi_embedded_text_input
        else:
            multi_encoded_text = None

        # Mono word (embedded_words)
        if self._mono_embedders:
            mono_embedder = self._mono_embedders[batch_tbid]
            embedded_words = mono_embedder(words)
            embedded_words = self._input_dropout_word(embedded_words)
        else:
            embedded_words = None

        # Mono token characters (encoded_token_characters)
        if self._mono_token_character_embedders:
            token_character_mask = util.get_text_field_mask(token_characters, num_wrapping_dims=1)
            batch_size, sequence_length, word_length = token_character_mask.size()

            # embed the characters
            mono_token_character_embedder = self._mono_token_character_embedders[batch_tbid]
            # Shape: (batch_size, sequence_length, word_length, embedding_dim)
            embedded_token_characters = mono_token_character_embedder(token_characters)

            # Shape: (batch_size * sequence_length, word_length, embedding_dim)
            embedded_token_characters = embedded_token_characters.view(batch_size * sequence_length, word_length, -1)
            embedded_token_characters = self._input_dropout_character(embedded_token_characters)

            # Shape: (batch_size * sequence_length, word_length)
            token_character_mask = token_character_mask.view(batch_size * sequence_length, word_length)

            # run the character LSTM (SEQ2VEC)
            if self._mono_token_character_encoders:
                mono_token_character_encoder = self._mono_token_character_encoders[batch_tbid]
                encoded_token_characters = mono_token_character_encoder(embedded_token_characters, token_character_mask)
                encoded_token_characters = encoded_token_characters.view(batch_size, sequence_length, -1)
                encoded_token_characters = self._dropout(encoded_token_characters)
        else:
            encoded_token_characters = None

        # Sentence-level character view
        if self._mono_sentence_character_embedders:
            sentence_character_mask = util.get_text_field_mask(sentence_characters)

            # embed the characters
            mono_sentence_character_embedder = self._mono_sentence_character_embedders[batch_tbid]
            embedded_sentence_characters = mono_sentence_character_embedder(sentence_characters)
            embedded_sentence_characters = self._input_dropout_character(embedded_sentence_characters)

            #print("scm", sentence_character_mask.size())
            #print("emsc", embedded_sentence_characters.size())

            # run the character LSTM
            if self._mono_sentence_character_encoders:
                mono_sentence_character_encoder = self._mono_sentence_character_encoders[batch_tbid]
                encoded_sentence_characters = mono_sentence_character_encoder(embedded_sentence_characters, sentence_character_mask)
                encoded_sentence_characters = self._dropout(encoded_sentence_characters)
                #print("ecsc", encoded_sentence_characters.size())
                #print(sentence_characters)
                word_starts_and_ends = self.find_word_start_and_end_indices(sentence_characters, batch_tbid, metadata)
                #print(sentence_characters)
                concatenated_word_start_and_end_representations = self.concatenate_word_start_and_end_indices(encoded_sentence_characters, word_starts_and_ends)
                padded_tensor_list = self.pad_character_representations(concatenated_word_start_and_end_representations)
                encoded_sentence_characters = self.convert_to_stacked_tensor(padded_tensor_list)
                ####
        else:
            encoded_sentence_characters = None

        # The above embedding/encoding steps optionally produce the following outputs:
        # * multi_encoded_text
        # * embedded_words
        # * encoded_token_characters
        # * encoded_sentence_characters
        
    
        if self._mono_encoders:
             # Pack the encoded text representations
            concatenated_input = []
            if embedded_words is not None:
                concatenated_input.append(embedded_words)
            if encoded_token_characters is not None:
                concatenated_input.append(encoded_token_characters)
            if encoded_sentence_characters is not None:
                concatenated_input.append(encoded_sentence_characters)
            if len(concatenated_input) > 1:
                text_features = torch.cat(concatenated_input, -1)
            else:
                text_features = concatenated_input[0]  
 
            mono_encoder = self._mono_encoders[batch_tbid]
            mono_encoded_text = mono_encoder(text_features, mask)
            mono_encoded_text = self._dropout(mono_encoded_text)

        else:
            # mono is a pre-trained transformer (which is just a text field embedder.)
            if embedded_words is not None:
                mono_encoded_text = embedded_words
            else:
                mono_encoded_text = None

        # Pack the encoded text representations
        encoded_text = {}
        if multi_encoded_text is not None:
            encoded_text["multi_encoded_text"] = multi_encoded_text
        if mono_encoded_text is not None:
            encoded_text["mono_encoded_text"] = mono_encoded_text

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
