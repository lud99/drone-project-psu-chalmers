package com.dji.sdk.sample;

import android.os.HandlerThread;
import android.util.Log;
import org.webrtc.IceCandidate; // WebRTC ICE Candidate
import org.json.JSONObject; // JSON handling
import org.json.JSONException; // JSON error handling
import androidx.annotation.Nullable;
import java.io.IOException;
import java.net.URI;
import java.util.concurrent.Semaphore;
import dev.gustavoavila.websocketclient.WebSocketClient;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;


import org.webrtc.CapturerObserver;
import org.webrtc.NV12Buffer;
import org.webrtc.SurfaceTextureHelper;
import org.webrtc.VideoCapturer;
import org.webrtc.VideoFrame;
import com.dji.sdk.sample.DJIVideoCapturer;  // Import DJIVideoCapturer (your custom implementation)
import dji.sdk.base.BaseProduct;
import dji.sdk.sdkmanager.DJISDKManager;



public class WebsocketClientHandler {
    // Delay between registration retries when telemetry is not ready, in milliseconds.
    private static final long REGISTRATION_RETRY_DELAY_MS = 500L;
    // Max number of retries for sending registration data when telemetry is not ready before giving up and sending with fallback data.
    private static final int MAX_REGISTRATION_RETRIES = 20;
    
    public interface ConnectionStateListener {
        void onConnected();
        void onDisconnected();
    }

    private ConnectionStateListener connectionStateListener;
    public void setConnectionStateListener(ConnectionStateListener listener) {
        this.connectionStateListener = listener;
    }
    private static WebsocketClientHandler clientHandler = null;
    private URI uri = null;
    private final WebSocketClient webSocketClient;
    public static final String TAG = WebsocketClientHandler.class.getName();
    private boolean connected = false;
    private String lastStringReceived = "Nothing received...";
    private byte[] lastBytesReceived = null;
    public static Semaphore new_string = new Semaphore(0);
    public static Semaphore status_update = new Semaphore(0);
    private final Context context;
    private WSPosition wsPositionRunnable;
    private Thread wsPositionThread;
    private WebRTCClient webRTCClient; 
    private DJIVideoCapturer DJIVideoCapturer; 
    private WebRTCMediaOptions webRTCMediaOptions;  
    private DroneAdapter droneAdapter;
    private boolean registrationDataSentForCurrentConnection = false;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private int registrationRetryCount = 0;
    private boolean registrationRetryScheduled = false;

    // Re-attempts registration from the main thread after a delayed retry window.
    private final Runnable registrationRetryRunnable = new Runnable() {
        @Override
        public void run() {
            synchronized (WebsocketClientHandler.this) {
                // Allow scheduleRegistrationRetryLocked to queue the next retry if needed.
                registrationRetryScheduled = false;
            }
            trySendRegistrationDataIfReady();
        }
    };
   
    


    /**
     * Get the active instance of the WebsocketClientHandler.
     * @return Returns null if the client hasn't been created, returns
     * the WebsocketClientHandler if it has been instantiated
     */

    @Nullable
    public static WebsocketClientHandler getInstance(){
        return clientHandler;
    }

    public static boolean isInstanceCreated(){
        return clientHandler != null;
    }

    public static WebsocketClientHandler createInstance(Context context, URI uri){
        clientHandler = new WebsocketClientHandler(context, uri);
        return clientHandler;
    }

    public WebSocketClient getWebSocketClient() {
        return webSocketClient;
    }

    public synchronized void refreshAdapterSelection() {
        this.droneAdapter = getSelectedAdapter();
    }

    public synchronized void onDroneDisconnected() {
        resetRegistrationRetryStateLocked();
        registrationDataSentForCurrentConnection = false;
        if (connected) {
            Log.i(TAG, "Drone disconnected; closing backend websocket connection.");
            closeConnection();
        }
    }

    public synchronized void trySendRegistrationDataIfReady() {
        Log.d(TAG, "xxxxx Attempting to send registration data if ready...");
        if (!connected || registrationDataSentForCurrentConnection) {
            return;
        }

        BaseProduct product = DJISDKManager.getInstance().getProduct();
        if (product == null) {
            return;
        }

        refreshAdapterSelection();
        if (droneAdapter == null) {
            return;
        }
        Log.d(TAG, "xxxxx Drone adapter found, requesting registration data asynchronously...");
        droneAdapter.getRegistrationData(new DroneAdapter.RegistrationDataCallback() {
            @Override
            public void onSuccess(DroneAdapter.RegistrationData registrationData) {
                if (registrationData == null) {
                    Log.w(TAG, "Registration data callback returned null; skipping send.");
                    return;
                }

                synchronized (WebsocketClientHandler.this) {
                    if (!connected || registrationDataSentForCurrentConnection) {
                        return;
                    }

                    DroneAdapter.Telemetry telemetry = droneAdapter.getTelemetry();
                    // Hold registration until telemetry looks initialized (not null/default values).
                    if (!isTelemetryReadyForRegistration(telemetry)) {
                        if (scheduleRegistrationRetryLocked("Telemetry not ready")) {
                            return;
                        }
                        Log.w(TAG, "Registration telemetry still unavailable after retries; sending registration with fallback telemetry payload.");
                    }

                    MessageHandler messageHandler = MessageHandler.getInstance();
                    if (messageHandler == null) {
                        return;
                    }

                    messageHandler.registrationData(registrationData, telemetry);
                    registrationDataSentForCurrentConnection = true;
                    resetRegistrationRetryStateLocked();
                    Log.i(TAG, "Registration data sent after async capabilities fetch completed.");
                }
            }

            @Override
            public void onFailure(String reason) {
                Log.w(TAG, "Registration data async fetch failed; skipping send. reason=" + reason);
            }
        });
    }

    private WebsocketClientHandler(Context context, URI uri){
        this.uri = uri;
        this.context = context;
        this.droneAdapter = getSelectedAdapter();
        webSocketClient = new WebSocketClient(uri) {
            @Override
            public void onOpen() {
                Log.d(TAG, "New connection opened on URI " + getUri());
                connected = true;
                registrationDataSentForCurrentConnection = false;
                resetRegistrationRetryStateLocked();
                if (connectionStateListener != null) {
                    connectionStateListener.onConnected();
                }

                // Run UI-related logic on the main thread
                new Handler(Looper.getMainLooper()).post(() -> {
                    if (webRTCClient == null) {
                        try {
                            Log.d(TAG, "Initializing WebRTCClient...");
                            VideoCapturer videoCapturer = new DJIVideoCapturer("DJI Mavic Enterprise 2");
                            WebRTCMediaOptions mediaOptions = new WebRTCMediaOptions();
                            webRTCClient = new WebRTCClient(context, videoCapturer, mediaOptions);
                            Log.d(TAG, "WebRTCClient initialized successfully.");
                        } catch (Exception e) {
                            Log.e(TAG, "Error initializing WebRTCClient: " + e.getMessage(), e);
                        }
                    } else {
                        Log.w(TAG, "WebRTCClient already initialized.");
                    }
                    trySendRegistrationDataIfReady();
                    startPositionSending(); // Ensure position sending starts properly
                    startHeartbeat();
                    WebsocketClientHandler.status_update.release();
                });
            }


            @Override
            public void onTextReceived(String message) {
                try {
                    JSONObject jsonMessage = new JSONObject(message);
                    handleIncomingMessage(jsonMessage, message);
                } catch (JSONException e) {
                    Log.e(TAG, "Failed to parse message: " + e.getMessage());
                }
            }

            @Override
            public void onBinaryReceived(byte[] data) {
                Log.d(TAG, "Received bytes");
                lastBytesReceived = data;
            }

            @Override
            public void onPingReceived(byte[] data) {
                Log.d(TAG, "PING");
            }

            @Override
            public void onPongReceived(byte[] data) {
                Log.d(TAG, "PONG");
            }

            @Override
            public void onException(Exception e) {
                Log.e(TAG, e.toString());
                connected = false;
                stopHeartbeat(); //Stop ping to backend
                stopPositionSending();
                WebsocketClientHandler.status_update.release();
            }

            @Override
            public void onCloseReceived(int reason, String description) {
                Log.d(TAG, String.format("Closed with code %d, %s", reason, description));
                connected = false;
                resetRegistrationRetryStateLocked();
                registrationDataSentForCurrentConnection = false;
                if (connectionStateListener != null) {
                    connectionStateListener.onDisconnected();
                }
                stopPositionSending();
                stopHeartbeat(); //Stop ping to backend
                if (webRTCClient != null) {
                    webRTCClient.dispose();
                    webRTCClient = null; // Nullify to prevent further usage
                    Log.d(TAG, "WebRTCClient disposed and nullified.");
                }
                WebsocketClientHandler.status_update.release();
        }

        };
        webSocketClient.setConnectTimeout(15000);
        webSocketClient.setReadTimeout(30000);
    }

    public URI getUri() {
        return uri;
    }

    public static WebsocketClientHandler resetClientHandler(Context context, URI uri) {
        clientHandler = new WebsocketClientHandler(context, uri);
        return clientHandler;
    }

    public boolean isConnected() {
        return connected;
    }

    public boolean send(String message){
        Log.w(TAG, "Sending...");
        if (isConnected()){
            webSocketClient.send(message);
            return true;
        } else{
            Log.e(TAG, "WebSocket is not connected.");
            return false;
        }
    }

    public boolean send(byte[] data){
        if (isConnected()){
            webSocketClient.send(data);
            return true;
        } else{
            Log.e(TAG, "WebSocket is not connected.");
            return false;
        }
    }

    public byte[] getLastBytesReceived() {
        return lastBytesReceived;
    }

    public String getLastStringReceived() {
        return lastStringReceived;
    }

    public void closeConnection(){
        connected = false;

        try {
            webSocketClient.close(0, 1001, "Connection closed by app");
        } catch (Exception e) {
            Log.d(TAG, "Failed to close connection!");
            e.printStackTrace();
        }
    }

    private void initializeWebRTCClient()
    {
        if (webRTCClient == null) {
            try {
                Log.d(TAG, "Initializing WebRTCClient...");
                VideoCapturer videoCapturer = new DJIVideoCapturer("DJI Mavic Enterprise 2");
                WebRTCMediaOptions mediaOptions = new WebRTCMediaOptions();
                webRTCClient = new WebRTCClient(context, videoCapturer, mediaOptions);
                Log.d(TAG, "WebRTCClient initialized successfully.");
            } catch (Exception e) {
                Log.e(TAG, "Error initializing WebRTCClient: " + e.getMessage(), e);
            }
        } else {
            Log.w(TAG, "WebRTCClient already initialized!!.");
        }
    }

    public boolean connect(){
        if (isConnected()){
            return false;
        }
        if (webSocketClient != null){
            refreshAdapterSelection();
            initializeWebRTCClient();
            webSocketClient.connect();

            return true;
        }
        return false;


    }

    private HandlerThread wsPositionHandlerThread;
    private Handler wsPositionHandler;
    
    private synchronized void startPositionSending() {
        if (wsPositionHandlerThread == null || !wsPositionHandlerThread.isAlive()) {
            Log.i(TAG, "Starting position sending HandlerThread...");
            wsPositionHandlerThread = new HandlerThread("WebSocketPositionSender");
            wsPositionHandlerThread.start();
            wsPositionHandler = new Handler(wsPositionHandlerThread.getLooper());
            wsPositionRunnable = new WSPosition(this.webSocketClient, droneAdapter);
    
            // Run WSPosition logic in the HandlerThread
            wsPositionHandler.post(wsPositionRunnable);
        } else {
            Log.w(TAG, "Position sending HandlerThread already running.");
        }
    }

    private DroneAdapter getSelectedAdapter() {
        return DroneAdapterManager.getCurrentAdapter();
    }

    private synchronized void resetRegistrationRetryStateLocked() {
        // Called when connection state changes or registration succeeds.
        registrationRetryCount = 0;
        registrationRetryScheduled = false;
        mainHandler.removeCallbacks(registrationRetryRunnable);
    }

    private synchronized boolean scheduleRegistrationRetryLocked(String reason) {
        if (registrationDataSentForCurrentConnection || !connected) {
            return false;
        }

        if (registrationRetryScheduled) {
            return true;
        }

        if (registrationRetryCount >= MAX_REGISTRATION_RETRIES) {
            Log.w(TAG, "Registration retry limit reached (" + MAX_REGISTRATION_RETRIES + "). reason=" + reason);
            return false;
        }

        registrationRetryCount++;
        registrationRetryScheduled = true;
        Log.i(TAG, "Delaying registration send; telemetry not ready. Retry " + registrationRetryCount + "/" + MAX_REGISTRATION_RETRIES + " in " + REGISTRATION_RETRY_DELAY_MS + " ms");
        mainHandler.postDelayed(registrationRetryRunnable, REGISTRATION_RETRY_DELAY_MS);
        return true;
    }

    private boolean isTelemetryReadyForRegistration(DroneAdapter.Telemetry telemetry) {
        if (telemetry == null) {
            return false;
        }

        // Treat default/uninitialized telemetry as not ready.
        // DJI often reports these values briefly right after connect before the first real fix.
        if (Double.isNaN(telemetry.lat) || Double.isNaN(telemetry.lon)) {
            return false;
        }

        return !(telemetry.lat == 0.0 && telemetry.lon == 0.0 && telemetry.batteryPercent == 0);
    }

    private void handleIncomingMessage(JSONObject jsonMessage, String rawMessage) {
        String type = jsonMessage.optString("msg_type", "");

        if ("Coordinate_request".equals(type)) {
            Log.d(TAG, "Received: " + rawMessage);
            lastStringReceived = rawMessage;
            new_string.release();
            return;
        }

        if ("offer".equals(type) || "candidate".equals(type) || "answer".equals(type)) {
            if (webRTCClient == null) {
                Log.w(TAG, "RTC CLIENT IS NULL, initializing...");
                initializeWebRTCClient();
            }
            if (webRTCClient != null) {
                webRTCClient.handleWebRTCMessage(jsonMessage);
            } else {
                Log.e(TAG, "RTC CLIENT still null after initialization, dropping message");
                
            }
            return;
        }

        if ("task".equals(type)) {
            handleTaskMessage(jsonMessage);
            return;
        }

        if ("abort_task".equals(type)) {
            refreshAdapterSelection();
            String missionId = readMissionId(jsonMessage);
            int taskIndex = readTaskIndex(jsonMessage);
            String taskType = jsonMessage.optString("task_action", "");
            if ("all".equals(taskType)) {
                droneAdapter.stopAllTasks(missionId, taskIndex);
            } else {
                droneAdapter.abortTask(missionId, taskIndex, taskType);
            }
            return;
        }

        if ("go_home".equals(type)) {
            refreshAdapterSelection();
            droneAdapter.goHome("manual", 0);
            return;
        }

        if ("land".equals(type)) {
            refreshAdapterSelection();
            droneAdapter.land("manual", 0);
            return;
        }

        Log.w(TAG, "Unhandled message type: " + type);
    }

    private void handleTaskMessage(JSONObject jsonMessage) {
        refreshAdapterSelection();
        if (droneAdapter == null) {
            Log.e(TAG, "No DroneAdapter selected when task message was received.");
            return;
        }

        String missionId = readMissionId(jsonMessage);
        int taskIndex = readTaskIndex(jsonMessage);

        JSONObject taskObject = jsonMessage.optJSONObject("task_action");
        if (taskObject == null) {
            taskObject = jsonMessage.optJSONObject("task");
        }
        String action;
        JSONObject params;

        if (taskObject != null) {
            action = taskObject.optString("action", "");
            params = taskObject.optJSONObject("params");
            if (params == null) {
                params = taskObject.optJSONObject("action_params");
            }
        } else {
            action = jsonMessage.optString("action", "");
            params = jsonMessage.optJSONObject("params");
            if (params == null) {
                params = jsonMessage.optJSONObject("action_params");
            }
        }

        if (params == null) {
            params = new JSONObject();
        }

        switch (action) {
            case "go_to":
                droneAdapter.goTo(
                        params.optDouble("lat", 0.0),
                        params.optDouble("lon", 0.0),
                        (float) params.optDouble("alt", 0.0),
                        params.has("heading") ? params.optInt("heading") : null,
                        missionId,
                        taskIndex
                );
                break;
            case "angle_camera":
                droneAdapter.angleCamera(
                        (float) params.optDouble("pitch", 0.0),
                        (float) params.optDouble("yaw", 0.0),
                        (float) params.optDouble("transition_time", 0.0),
                        missionId,
                        taskIndex
                );
                break;
            case "play_audio":
                droneAdapter.playAudio(
                        params.optString("file", ""),
                        (float) params.optDouble("volume", 1.0),
                        params.has("duration_seconds") ? params.optInt("duration_seconds") : null,
                        missionId,
                        taskIndex
                );
                break;
            case "stop_audio":
                droneAdapter.stopAudio(missionId, taskIndex);
                break;
            case "led":
                droneAdapter.led(
                        params.optString("type", params.optString("color", "beacon")),
                        params.has("duration_seconds") ? params.optInt("duration_seconds") : null,
                        missionId,
                        taskIndex
                );
                break;
            case "deactivate_led":
                droneAdapter.deactivateLed(
                        params.optString("type", params.optString("color", "beacon")),
                        missionId,
                        taskIndex
                );
                break;
            case "spotlight":
                droneAdapter.spotlight(
                        (float) params.optDouble("brightness", 1.0),
                        params.has("duration_seconds") ? params.optInt("duration_seconds") : null,
                        missionId,
                        taskIndex
                );
                break;
            case "deactivate_spotlight":
                droneAdapter.deactivateSpotlight(missionId, taskIndex);
                break;
            case "go_home":
                droneAdapter.goHome(missionId, taskIndex);
                break;
            case "land":
                droneAdapter.land(missionId, taskIndex);
                break;
            case "abort_all_tasks":
                droneAdapter.stopAllTasks(missionId, taskIndex);
                break;
            case "abort_task":
                droneAdapter.abortTask(
                        missionId,
                        taskIndex,
                        params.optString("task_type", params.optString("taskType", ""))
                );
                break;
            default:
                Log.w(TAG, "Unsupported task action: " + action);
                break;
        }
    }

    private String readMissionId(JSONObject jsonMessage) {
        String missionId = jsonMessage.optString("mission_id", "");
        if (missionId.isEmpty()) {
            missionId = jsonMessage.optString("missionID", "");
        }
        if (missionId.isEmpty()) {
            missionId = "unknown";
        }
        return missionId;
    }

    private int readTaskIndex(JSONObject jsonMessage) {
        if (jsonMessage.has("index")) {
            return jsonMessage.optInt("index", 0);
        }
        if (jsonMessage.has("task_index")) {
            return jsonMessage.optInt("task_index", 0);
        }
        if (jsonMessage.has("taskIndex")) {
            return jsonMessage.optInt("taskIndex", 0);
        }
        return 0;
    }
    
    private synchronized void stopPositionSending() {
        if (wsPositionHandlerThread != null) {
            Log.i(TAG, "Stopping position sending HandlerThread...");
            wsPositionHandlerThread.quitSafely(); // Quit the HandlerThread gracefully
            try {
                wsPositionHandlerThread.join(); // Wait for it to finish
                Log.i(TAG, "HandlerThread stopped.");
            } catch (InterruptedException e) {
                Log.w(TAG, "Interrupted while stopping HandlerThread.");
                Thread.currentThread().interrupt();
            }
            wsPositionRunnable = null; // Clear references
            wsPositionHandlerThread = null;
            wsPositionHandler = null;
        }
    }


private Handler heartbeatHandler = null; //Sending pings to backend to check if it's still connected

private synchronized void startHeartbeat() { 
    stopHeartbeat();
    heartbeatHandler = new Handler(Looper.getMainLooper());
    heartbeatHandler.postDelayed(heartbeatRunnable, 5000); //Sending ping every 5 seconds
}

private synchronized void stopHeartbeat() {
    if (heartbeatHandler != null) {
        heartbeatHandler.removeCallbacks(heartbeatRunnable);
        heartbeatHandler = null;
    }
}



private final Runnable heartbeatRunnable = new Runnable() {
    @Override
    public void run() {
        if (!connected) return;

        try {
            String ping = "{\"msg_type\": \"ping\"}";
            webSocketClient.send(ping);
            Log.d(TAG, "Heartbeat ping sent");
            if (heartbeatHandler != null) {
                heartbeatHandler.postDelayed(this, 5000);
            }
        } catch (Exception e) {
            Log.e(TAG, "Heartbeat failed: " + e.getMessage());
            connected = false;
            stopHeartbeat();
            status_update.release();
        }
    }
};

}

