package com.dji.sdk.sample;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import dji.sdk.media.AudioMediaFile;
import dji.sdk.products.Aircraft;


/**
 * AudioFileMapping is a utility class that provides methods to check for speaker capabilities
 * and to build a mapping of audio file names to their corresponding indices for DJI drones.
 */
public final class AudioFileMapping {

    private AudioFileMapping() {
    }

    public static boolean hasSpeaker(Aircraft aircraft) {
        return aircraft != null
                && aircraft.getAccessoryAggregation() != null
                && aircraft.getAccessoryAggregation().getSpeaker() != null;
    }

    // This method builds the speaker capabilities and caches the mapping of audio file names to their indices.
    // Returns a Capabilities.Speaker object containing the list of audio file names, and fills the provided cache map with name-index pairs.
    public static DroneAdapter.Capabilities.Speaker buildSpeakerCapabilitiesAndCache(
            Aircraft aircraft,
            Map<String, Integer> audioIndexCache
    ) {
        if (audioIndexCache != null) {
            audioIndexCache.clear();
        }

        if (!hasSpeaker(aircraft)) {
            return null;
        }

        DroneAdapter.Capabilities.Speaker speakerCapabilities = new DroneAdapter.Capabilities.Speaker();

        List<AudioMediaFile> audioFiles = aircraft.getAccessoryAggregation().getSpeaker().getFileListSnapshot();
        List<String> audioFileNames = new ArrayList<>();

        if (audioFiles != null) {
            for (AudioMediaFile file : audioFiles) {
                String fileName = file.getFileName();
                int fileIndex = file.getIndex();

                audioFileNames.add(fileName);
                if (audioIndexCache != null) {
                    audioIndexCache.put(fileName, fileIndex);
                }
            }
        }

        speakerCapabilities.audio_files = audioFileNames.toArray(new String[0]);
        return speakerCapabilities;
    }

    // This method retrieves the cached index for a given audio file name from the cache map.
    // Returns the index if found, or null if the file name is not in the cache.
    public static Integer getCachedFileIndex(Map<String, Integer> audioIndexCache, String fileName) {
        if (audioIndexCache == null || fileName == null) {
            return null;
        }
        return audioIndexCache.get(fileName);
    }
}
