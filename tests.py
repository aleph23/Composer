import pytest
import numpy as np
from midi_utils import midi_to_samples, samples_to_midi
from mido import MidiFile
import os

# Mock params module as it's not provided
class MockParams:
    encode_volume = True
    encode_length = True

params = MockParams()

@pytest.mark.parametrize("file_name, num_notes, samples_per_measure, expected_exception, test_id", [
    ("test_midi_1.mid", 96, 96, None, "happy_path_basic"),
    ("test_midi_2.mid", 128, 48, None, "happy_path_extended_notes"),
    ("test_midi_invalid.mid", 96, 96, NotImplementedError, "error_multiple_time_signatures"),
    ("test_midi_nonexistent.mid", 96, 96, FileNotFoundError, "error_file_not_found"),
])
def test_midi_to_samples(file_name, num_notes, samples_per_measure, expected_exception, test_id, tmpdir):
    if expected_exception:
        with pytest.raises(expected_exception):
            midi_to_samples(file_name, num_notes, samples_per_measure)
    else:
        # Arrange
        midi_path = os.path.join(tmpdir, file_name)
        MidiFile().save(midi_path)  # Create a simple, empty MIDI file for testing

        # Act
        samples = midi_to_samples(midi_path, num_notes, samples_per_measure)

        # Assert
        assert isinstance(samples, list), f"Test ID {test_id}: The result should be a list."
        if samples:  # If there are samples, check their structure
            assert isinstance(samples[0], np.ndarray), f"Test ID {test_id}: Each sample should be a numpy array."
            assert samples[0].shape == (samples_per_measure, num_notes), f"Test ID {test_id}: Incorrect shape of sample."

@pytest.mark.parametrize("samples, file_name, threshold, num_notes, samples_per_measure, expected_notes, test_id", [
    ([np.zeros((96, 96))], "output_1.mid", 0.5, 96, 96, 0, "empty_samples"),
    ([np.ones((96, 96)) * 0.6], "output_2.mid", 0.5, 96, 96, 96, "full_samples_above_threshold"),
    ([np.ones((96, 96)) * 0.4], "output_3.mid", 0.5, 96, 96, 0, "full_samples_below_threshold"),
])
def test_samples_to_midi(samples, file_name, threshold, num_notes, samples_per_measure, expected_notes, test_id, tmpdir):
    # Arrange
    output_path = os.path.join(tmpdir, file_name)

    # Act
    samples_to_midi(samples, output_path, threshold, num_notes, samples_per_measure)

    # Assert
    mid = MidiFile(output_path)
    note_on_messages = sum(1 for track in mid.tracks for msg in track if msg.type == 'note_on')
    assert note_on_messages == expected_notes, f"Test ID {test_id}: Expected {expected_notes} 'note_on' messages, found {note_on_messages}."
