package com.dji.sdk.sample;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import dji.sdk.media.AudioMediaFile;
import dji.sdk.products.Aircraft;

public final class AudioFileMapping {

    private AudioFileMapping() {
    }

    public static boolean hasSpeaker(Aircraft aircraft) {
        return aircraft != null
                && aircraft.getAccessoryAggregation() != null
                && aircraft.getAccessoryAggregation().getSpeaker() != null;
    }

    public static DroneAdapter.Capabilities.Speaker buildSpeakerCapabilitiesAndCache(
            Aircraft aircraft,
            Map<String, Integer> audioIndexCache
    ) {
        DroneAdapter.Capabilities.Speaker speakerCapabilities = new DroneAdapter.Capabilities.Speaker();

        if (audioIndexCache != null) {
            audioIndexCache.clear();
        }

        if (!hasSpeaker(aircraft)) {
            speakerCapabilities.audio_files = new String[0];
            return speakerCapabilities;
        }

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

    public static Integer getCachedFileIndex(Map<String, Integer> audioIndexCache, String fileName) {
        if (audioIndexCache == null || fileName == null) {
            return null;
        }
        return audioIndexCache.get(fileName);
    }
}
