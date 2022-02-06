import unittest
import torch


class TestWordStartEnd(unittest.TestCase):
    """Test the logic for gathering the indices of the starts and ends of words."""

    def test_multiple_word_start_and_end_indices(self):
        """Try a large number of samples to see if anything goes wrong."""

        batch_size = 32
        seq_len = 80
        space_index = 2
        padding_index = 0

        for i in range(10000):
            # each batch element contains seq_len characters
            character_tensor = torch.LongTensor(batch_size, seq_len).random_(0, 10)

            start_locations = []
            end_locations = []
            for sentence in character_tensor:
                have_seen_space_char = False
                start_of_sentence = True
                sentence_starts = []
                sentence_ends = []
                for i, value in enumerate(sentence):
                    if start_of_sentence or last_was_space_char:
                        sentence_starts.append(i)
                        start_of_sentence = False
                        last_was_space_char = False
                    else:
                        if value == space_index:
                            end_of_word_idx = i - 1
                            sentence_ends.append(end_of_word_idx)
                            last_was_space_char = True
                            have_seen_space_char = True
                        elif value == padding_index:
                            end_of_word_idx = i - 1
                            sentence_ends.append(end_of_word_idx)
                            break

                if not have_seen_space_char:
                    sentence_ends.append(i)

                if len(sentence_starts) > len(sentence_ends):
                    assert i == (len(sentence) - 1)
                    sentence_ends.append(i)

                if len(sentence_starts) >= 1 and len(sentence_ends) >= 1:
                    start_locations.append(sentence_starts)
                    end_locations.append(sentence_ends)
                else:
                    raise ValueError(
                        "Something has gone wrong with the indexing, no indices were found."
                    )

            location_tuples = []
            for sentence_starts, sentence_ends in zip(start_locations, end_locations):
                current_sent = []
                for word_start, word_end in zip(sentence_starts, sentence_ends):
                    word_start_and_end = (word_start, word_end)
                    current_sent.append(word_start_and_end)
                location_tuples.append(current_sent)

            assert len(location_tuples) == character_tensor.size(0), print(
                f"collected {len(location_tuples)} location tuples, \
                                                                        for a tensor with {character_tensor.size(0)} elements"
            )

            return location_tuples

    def test_word_start_and_end_indices(self):
        """Try a single example to see if anything goes wrong."""

        space_index = 2
        padding_index = 0
        num_space_chars = 0

        # each batch element contains seq_len characters
        character_tensor = torch.tensor(
            [
                [16, 14, 2, 6, 5, 4, 3, 2, 11, 0],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        start_locations = []
        end_locations = []
        for sentence in character_tensor:
            have_seen_space_char = False
            start_of_sentence = True
            sentence_starts = []
            sentence_ends = []
            for i, char in enumerate(sentence):
                if start_of_sentence or last_was_space_char:
                    sentence_starts.append(i)
                    start_of_sentence = False
                    last_was_space_char = False
                else:
                    if char == space_index:
                        num_space_chars += 1
                        end_of_word_idx = i - 1
                        sentence_ends.append(end_of_word_idx)
                        last_was_space_char = True
                        have_seen_space_char = True
                    elif char == padding_index:
                        end_of_word_idx = i - 1
                        sentence_ends.append(end_of_word_idx)
                        break

            if not have_seen_space_char:
                sentence_ends.append(i)

            if len(sentence_starts) > len(sentence_ends):
                assert i == (len(sentence) - 1)
                sentence_ends.append(i)

            if len(sentence_starts) >= 1 and len(sentence_ends) >= 1:
                start_locations.append(sentence_starts)
                end_locations.append(sentence_ends)
            else:
                raise ValueError(
                    "Something has gone wrong with the indexing, no indices were found."
                )

        # 3 space chars in this sample batch
        assert num_space_chars == 3

        location_tuples = []
        for sentence_starts, sentence_ends in zip(start_locations, end_locations):
            current_sent = []
            for word_start, word_end in zip(sentence_starts, sentence_ends):
                word_start_and_end = (word_start, word_end)
                current_sent.append(word_start_and_end)
            location_tuples.append(current_sent)

        print(
            location_tuples
        )  # [[(0, 1), (3, 6), (8, 8)], [(0, 0), (2, 9)], [(0, 0)], [(0, 0)]]

        assert len(location_tuples) == character_tensor.size(0), print(
            f"collected {len(location_tuples)} location tuples, \
                                                                    for a tensor with {character_tensor.size(0)} elements"
        )

        return location_tuples
