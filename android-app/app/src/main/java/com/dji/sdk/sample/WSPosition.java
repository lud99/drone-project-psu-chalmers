package com.dji.sdk.sample;

import android.util.Log;

import java.util.Locale;

import dev.gustavoavila.websocketclient.WebSocketClient;

class WSPosition implements Runnable {
    private static final String TAG = WSPosition.class.getSimpleName();
    private final WebSocketClient webSocketClient;
    private final DroneAdapter droneAdapter;
    private volatile boolean isRunning = true; // Flag to control the loop

    // Constructor to receive the WebSocketClient instance
    public WSPosition(WebSocketClient client, DroneAdapter droneAdapter) {
        this.webSocketClient = client;
        this.droneAdapter = droneAdapter;
    }

    @Override
    public void run() {
        Log.i(TAG, "WSPosition thread started.");
        // TODO Check that it is connected
        while (isRunning && webSocketClient != null ) {
            try {
                if (droneAdapter == null) {
                    Log.e(TAG, "DroneAdapter is null.");
                    Thread.sleep(1000);
                    continue;
                }

                DroneAdapter.Telemetry telemetry = droneAdapter.getTelemetry();

                if (telemetry != null) {
                    String message = String.format(Locale.US,
                            "{\"msg_type\": \"Telemetry\",\"droneID\": \"%s\", \"lat\": %.8f, \"lon\": %.8f, \"alt\": %.2f, \"heading\": %.2f, \"speed\": %.2f, \"batteryPercent\": %d}",
                            telemetry.droneID, telemetry.lat, telemetry.lon, telemetry.alt, telemetry.heading, telemetry.speed, telemetry.batteryPercent);

                    Log.d(TAG, "Sending telemetry: " + message);
                    webSocketClient.send(message);
                } else {
                    Log.d(TAG, "Waiting for valid telemetry data...");
                }

                // Wait for 1 second before sending the next update
                Thread.sleep(1000);

            } catch (InterruptedException e) {
                Log.w(TAG, "WSPosition thread interrupted.");
                Thread.currentThread().interrupt(); // Restore interruption status
                isRunning = false; // Stop the loop if interrupted
            } catch (Exception e) {
                // Catch other potential exceptions during data fetching or sending
                Log.e(TAG, "Error in WSPosition loop: " + e.getMessage(), e);
                // Consider adding a small delay before retrying after an error
                try { Thread.sleep(500); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); isRunning = false; }
            }
        }
        Log.i(TAG, "WSPosition thread finished.");
    }

    // Method to signal the thread to stop
    public void stopRunning() {
        isRunning = false;
        Log.i(TAG, "WSPosition stop requested.");
    }
} 
