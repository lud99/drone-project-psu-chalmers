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



public class WebsocketClientHandler {
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



    private WebsocketClientHandler(Context context, URI uri){
        this.uri = uri;
        this.context = context;
        webSocketClient = new WebSocketClient(uri) {
            @Override
            public void onOpen() {
                Log.d(TAG, "New connection opened on URI " + getUri());
                connected = true;

                // Run UI-related logic on the main thread
                new Handler(Looper.getMainLooper()).post(() -> {
                    startPositionSending(); // Ensure position sending starts properly
                    WebsocketClientHandler.status_update.release();
                });
            }

            @Override
            public void onTextReceived(String message) {
                try {
                    JSONObject jsonMessage = new JSONObject(message);
                    String type = jsonMessage.getString("msg_type");
                    // FlightManager flightManager = FlightManager.getFlightManager();

                    if (type.equals("Coordinate_request")) {
                        Log.d(TAG, "Received: " + message);
                        lastStringReceived = message;
                        new_string.release();
                    } else if (type.equals("flight_arm")) {
                        Log.d(TAG, "Attempting to take off");
                    FlightManager flightManager = FlightManager.getFlightManager();
                        flightManager.onArm();
                    } else if (type.equals("offer") || type.equals("candidate") || type.equals("answer")) {
                        if(webRTCClient == null)
                        {
                            Log.d(TAG, "RTC CLIENT IS NULL");
                        }
                        webRTCClient.handleWebRTCMessage(jsonMessage);
                    } else if (type.equals("flight_take_off")) {
                        Log.d(TAG, "Attempting to take off");
                    FlightManager flightManager = FlightManager.getFlightManager();
                        flightManager.startWaypointMission();
                    } else if (type.equals("flight_return_to_home")) {
                        Log.d(TAG, "Attempting to return to home");
                    FlightManager flightManager = FlightManager.getFlightManager();
                        flightManager.goingHome();
                    } else {
                        Log.w(TAG, "Unhandled message type: " + type);
                    }
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
                if (e instanceof IOException){
                    //closeConnection();
                }
                WebsocketClientHandler.status_update.release(); //?
            }

            @Override
            public void onCloseReceived(int reason, String description) {
                Log.d(TAG, String.format("Closed with code %d, %s", reason, description));
                connected = false;
                stopPositionSending();
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
        //webSocketClient.enableAutomaticReconnection(1000);
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
            wsPositionRunnable = new WSPosition(this.webSocketClient);
    
            // Run WSPosition logic in the HandlerThread
            wsPositionHandler.post(wsPositionRunnable);
        } else {
            Log.w(TAG, "Position sending HandlerThread already running.");
        }
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

}

