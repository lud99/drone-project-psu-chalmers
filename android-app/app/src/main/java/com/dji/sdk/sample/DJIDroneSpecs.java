package com.dji.sdk.sample;

import android.util.Log;

import dji.common.camera.SettingsDefinitions;
import dji.common.camera.SettingsDefinitions.VideoFov;
import dji.common.camera.SettingsDefinitions.VideoResolution;
import dji.common.product.Model;


/**
 * DJIDroneSpecs is a utility class that provides methods to retrieve specifications of DJI drones,
 * such as camera resolution and field of view, based on the drone model and camera settings.
 */
public class DJIDroneSpecs {

    public static class Resolution {
        public int width;
        public int height;

        public Resolution(int w, int h) {
            this.width = w;
            this.height = h;
        }
    }

    /**
     * Parses width and height from DJI's VideoResolution enum.
     * Handles formats like RESOLUTION_1920x1080 and RESOLUTION_5760X3240.
     */
    public static Resolution getDimensions(VideoResolution resolution) {
        if (resolution == null || resolution == VideoResolution.UNKNOWN) {
            return new Resolution(1920, 1080); // Default fallback
        }

        try {
            String resName = resolution.name();
            if (resName.startsWith("RESOLUTION_")) {
                // Remove the prefix
                String parts = resName.replace("RESOLUTION_", "");
                
                // Split at 'x' or 'X'
                String[] dimensions = parts.split("[xX]");
                
                if (dimensions.length >= 2) {
                    int width = Integer.parseInt(dimensions[0]);
                    int height = Integer.parseInt(dimensions[1]);
                    return new Resolution(width, height);
                }
            }
        } catch (Exception e) {
            // If something goes wrong during parsing (e.g., RESOLUTION_MAX)
            Log.e("DJIDroneSpecs", "Error parsing resolution: " + e.getMessage());
        }

        return new Resolution(1920, 1080); // Fallback
    }

    /**
     * Map Model to horizontal FOV.
     * All the data was gathered using Gemini Deep Research.
     * Some drones have focal length adjustment, for those the max FOV is returned
     * since this will be the default
     * Contains all models listed in DJI's official documentation as of 2026-02
     * https://developer.dji.com/api-reference/android-api/BaseClasses/DJIBaseProduct.html#djimodelnamestring_inline
     */
    public static double getHorizontalFov(Model model, VideoFov vFov) {
        double baseFov = 84.0;
        if (model == null) {
            return baseFov;
        }

        String modelName = model.name();
        double modelFov;

        switch (modelName) {
            case "UNKNOWN_AIRCRAFT":
                modelFov = baseFov;
                break;
            case "INSPIRE_1":
                modelFov = 81.9;
                break;
            case "INSPIRE_1_PRO":
                modelFov = 62.1;
                break;
            case "INSPIRE_1_RAW":
                modelFov = 62.1;
                break;
            case "INSPIRE_2":
                modelFov = 73.7;
                break;
            case "PHANTOM_3_PROFESSIONAL":
                modelFov = 81.9;
                break;
            case "PHANTOM_3_ADVANCED":
                modelFov = 81.9;
                break;
            case "PHANTOM_3_STANDARD":
                modelFov = 81.9;
                break;
            case "PHANTOM_3_4K":
                modelFov = 81.9;
                break;
            case "PHANTOM_4":
                modelFov = 81.9;
                break;
            case "PHANTOM_4_PRO":
                modelFov = 73.7;
                break;
            case "PHANTOM_4_PRO_V2":
                modelFov = 73.7;
                break;
            case "P_4_MULTISPECTRAL":
                modelFov = 54.1;
                break;
            case "MAVIC_AIR_2":
                modelFov = 73.7;
                break;
            case "DJI_MINI_SE":
                modelFov = 72.7;
                break;
            case "DJI_MINI_2":
                modelFov = 72.7;
                break;
            case "MAVIC_2_ENTERPRISE_ADVANCED":
                modelFov = 73.7;
                break;
            case "MATRICE_100":
                modelFov = 81.9; 
                break;
            case "MATRICE_600":
                modelFov = 81.9;
                break;
            case "MATRICE_600_PRO":
                modelFov = 81.9;
                break;
            case "A3":
                modelFov = 50.0; 
                break;
            case "MAVIC_PRO":
                modelFov = 69.1;
                break;
            case "SPARK":
                modelFov = 71.6;
                break;
            case "MATRICE_210":
                modelFov = 73.7;
                break;
            case "MATRICE_210_RTK":
                modelFov = 73.7;
                break;
            case "MATRICE_200_V2":
                modelFov = 73.7;
                break;
            case "MATRICE_210_V2":
                modelFov = 73.7;
                break;
            case "MATRICE_210_RTK_V2":
                modelFov = 73.7;
                break;
            case "MATRICE_300_RTK":
                modelFov = 54.0; 
                break;
            case "MAVIC_AIR":
                modelFov = 74.6;
                break;
            case "MAVIC_2_PRO":
                modelFov = 68.4;
                break;
            case "MAVIC_2_ZOOM":
                modelFov = 72.7;
                break;
            case "MAVIC_2":
                modelFov = 68.4; 
                break;
            case "MAVIC_2_ENTERPRISE":
                modelFov = 72.4;
                break;
            case "MAVIC_2_ENTERPRISE_DUAL":
                modelFov = 74.6; 
                break;
            case "MAVIC_MINI":
                modelFov = 72.7;
                break;
            case "DJI_AIR_2S":
                modelFov = 77.6;
                break;
            case "N3":
                modelFov = 50.0; 
                break;
            case "UNKNOWN_HANDHELD":
                modelFov = 0.0;
                break;
            case "OSMO":
                modelFov = 81.9;
                break;
            case "OSMO_PRO":
                modelFov = 62.1;
                break;
            case "OSMO_RAW":
                modelFov = 62.1;
                break;
            case "OSMO_MOBILE":
                modelFov = 0.0; 
                break;
            case "OSMO_PLUS":
                modelFov = 80.0;
                break;
            case "OSMO_MOBILE_2":
                modelFov = 0.0;
                break;
            default:
                modelFov = baseFov;
                break;
        }

        if (vFov == VideoFov.NARROW && modelFov > 0.0) {
            return modelFov * 0.58;
        }
        return modelFov;
    }
}