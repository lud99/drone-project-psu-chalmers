package com.dji.sdk.sample;

import android.util.Log;

import org.json.JSONObject;

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
        while (isRunning && webSocketClient != null ) {
            try {
                if (droneAdapter == null) {
                    Log.e(TAG, "DroneAdapter is null.");
                    Thread.sleep(1000);
                    continue;
                }

                DroneAdapter.Telemetry telemetry = droneAdapter.getTelemetry();

                if (telemetry != null) {
                    JSONObject telemetryJson = new JSONObject();
                    telemetryJson.put("lat", telemetry.lat);
                    telemetryJson.put("lon", telemetry.lon);
                    telemetryJson.put("alt", telemetry.alt);
                    telemetryJson.put("heading", telemetry.heading);
                    telemetryJson.put("speed", telemetry.speed);
                    telemetryJson.put("battery_percent", telemetry.batteryPercent);

                    JSONObject messageJson = new JSONObject();
                    messageJson.put("msg_type", "telemetry");
                    messageJson.put("drone_id", telemetry.droneID);
                    messageJson.put("telemetry", telemetryJson);

                    String message = messageJson.toString();

                    Log.d(TAG, "Sending telemetry: " + message);
                    webSocketClient.send(message);
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
