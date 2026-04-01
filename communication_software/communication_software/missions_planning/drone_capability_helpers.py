"""
Module for handling the speaker component of a drone.
The speaker has a list of audio files that can be played.
Each audio file belongs to a specific type (e.g. alert, greeting, intruder instructions, stray car).
"""

# Mapping from audio types to audio files.
# The order in the list determines what audio file should be used if
# only the audio type is specified in the mission parameters.

audio_file_mapping = {
    "alert": ["horn", "siren"],
    "greeting": ["hello", "hi", "welcome"],
    "intruder_instructions": ["leave_track", "stay"],
    "stray_car": ["restart_transponder", "go_home"],
}


class AudioFile:
    """Represents an audio file that can be played by the drone's speaker."""

    def _map_audio_file(self, audio_file: str) -> str:
        """Maps an audio file to its corresponding audio type based on the audio_file_mapping."""
        for key, values in audio_file_mapping.items():
            if audio_file in values:
                return key
        raise NoCategoryError(
            f"Audio file {audio_file} does not belong to any category."
        )

    def __init__(self, audio_file: str):
        self.audio_file = audio_file
        self.audio_type = self._map_audio_file(audio_file)


class NoCategoryError(Exception):
    """Raised when an audio file does not belong to any category in the mapping."""


class SpeakerHelper:
    """Represents the speaker component of a drone, which can play different audio files."""

    audio_files: list[AudioFile]

    def __init__(self, audio_files: list[str]):
        self.audio_files = [AudioFile(audio_file) for audio_file in audio_files]

    def has_files(self) -> bool:
        """Returns True if the speaker has any audio files, False otherwise."""
        return len(self.audio_files) > 0

    def sort_key(self, audio_file: AudioFile):
        """Sorts audio files based on their type and the order in the mapping."""
        if audio_file.audio_type is None:
            raise NoCategoryError(
                f"Audio file {audio_file.audio_file} does not belong to any category."
            )
        mapped_list = audio_file_mapping[audio_file.audio_type]
        return mapped_list.index(audio_file.audio_file)

    def get_files_by_type(self, audio_type: str) -> list[AudioFile]:
        """
        Returns a list of audio files of the specified type,
        sorted by their order in the mapping.
        """
        filtered_files = [
            audio_file
            for audio_file in self.audio_files
            if audio_file.audio_type == audio_type
        ]
        return sorted(filtered_files, key=self.sort_key)

    def get_single_file_by_type(self, audio_type: str) -> AudioFile | None:
        """
        Returns a single audio file of the specified type.
        If multiple files of the same type exist, the one with the highest priority is returned.
        If no files of the specified type exist, None is returned.
        """
        files_by_type = self.get_files_by_type(audio_type)
        return files_by_type[0] if files_by_type else None

    def get_all_files(self) -> list[AudioFile]:
        """Returns a list of all audio files, sorted by their type and order in the mapping."""
        return self.audio_files

    def has_type(self, audio_type: str) -> bool:
        """Returns True if the speaker has any audio files of the specified type."""
        return any(
            audio_file.audio_type == audio_type for audio_file in self.audio_files
        )

    def has_file(self, audio_file: str) -> bool:
        """Returns True if the speaker has the specified audio file."""
        return any(f.audio_file == audio_file for f in self.audio_files)


class LEDHelper:
    """Class representing the LED system of a drone."""

    def __init__(self, led_types: list[str]) -> None:
        self.front = "front" in led_types
        self.rear = "rear" in led_types
        self.beacon = "beacon" in led_types

    def has_any(self) -> bool:
        """Returns True if any LED type is available."""
        return self.front or self.rear or self.beacon

    def has_type(self, led_type: str) -> bool:
        """Returns True if the requested LED type is available."""
        return {
            "front": self.front,
            "rear": self.rear,
            "beacon": self.beacon,
        }.get(led_type, False)
