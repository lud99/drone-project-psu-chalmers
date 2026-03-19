package com.dji.sdk.sample;

import android.util.Log;

import androidx.annotation.Nullable;

import dji.sdk.base.BaseProduct;


/*
* Selects which DroneAdapter implementation to use based on connected product or drone type.
* It currently defaults to DJIAdapter, but is designed to be extended for additional drone types.
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
                // Additional drone types can be added here when implemented.
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
