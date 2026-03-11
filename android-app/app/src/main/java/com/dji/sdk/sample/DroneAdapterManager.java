package com.dji.sdk.sample;

import android.util.Log;

import androidx.annotation.Nullable;

import dji.sdk.base.BaseProduct;


/*
* Manages the selection of the version of DroneAdapter to use based on the connected product or specified drone type.
* Currently, it defaults to DJIAdapter for all cases, but it is designed to be easily extendable to support multiple drone types in the future.
*/
public final class DroneAdapterManager {
    private static final String TAG = DroneAdapterManager.class.getSimpleName();
    private static DroneAdapter currentAdapter = DJIAdapter.getInstance();

    private DroneAdapterManager() {
    }

    public static synchronized void selectAdapterForConnectedProduct(@Nullable BaseProduct product) {
        if (product != null) {
            currentAdapter = DJIAdapter.getInstance();
            Log.i(TAG, "Connected product detected via DJI SDK. Selected adapter: DJIAdapter");
            return;
        }

        currentAdapter = DJIAdapter.getInstance();
        Log.w(TAG, "No connected product detected. Falling back to default adapter: DJIAdapter");
    }

    public static synchronized void selectAdapterByDroneType(String droneType) {
        if (droneType == null) {
            currentAdapter = DJIAdapter.getInstance();
            Log.w(TAG, "Drone type is null. Falling back to DJIAdapter");
            return;
        }

        String normalizedType = droneType.trim().toLowerCase();
        switch (normalizedType) {
            case "dji":
                currentAdapter = DJIAdapter.getInstance();
                break;
            case "mavlink":
                // When implementing new drone tyepes, these can get added here.
                currentAdapter = DJIAdapter.getInstance();
                Log.w(TAG, "MavlinkAdapter not implemented yet. Falling back to DJIAdapter");
                break;
            default:
                currentAdapter = DJIAdapter.getInstance();
                Log.w(TAG, "Unknown drone type '" + droneType + "'. Falling back to DJIAdapter");
                break;
        }
    }

    public static synchronized DroneAdapter getCurrentAdapter() {
        return currentAdapter;
    }
}
