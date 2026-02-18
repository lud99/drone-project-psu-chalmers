package com.dji.sdk.sample;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList; 

public class AutoConnectManager {
    //TAG, INFORMATION
    private static final String TAG = AutoConnectManager.class.getName(); //TAG CLASS NAME
    private static final long CHECK_INTERVAL_MS = 15000; // 15 seconds 
    private static final String PREFS_NAME = "ConnectionPrefs";
    private static final String PREF_LAST_IP = "LAST_IP";
    private static final String PREF_LAST_PORT = "LAST_PORT";

    // Only one of
    private static AutoConnectManager instance;

    //COMPONENTS
    private final Context context;
    private final MulticastReceiver multicastReceiver;
    private final HandlerThread handlerThread;
    private final Handler handler;

    //Initial STATE
    private String manualIp = null;
    private Integer manualPort = null;
    private boolean isManualOverride = false;
    private boolean isRunning = false;
    private boolean connecting = false; 


    //Connection every 15 sec 
    private final Runnable checkConnectionRunnable = new Runnable() {
        @Override
        public void run() {
            if (isRunning) {
                checkAndConnect();
                handler.postDelayed(this, CHECK_INTERVAL_MS);    
            }
        }
    };


    //Creates multicast receiver and opens port
    private AutoConnectManager(Context context) {
        this.context = context.getApplicationContext();
        this.multicastReceiver = new MulticastReceiver(9992);

        //Creates handler and starts it
        handlerThread = new HandlerThread("AutoConnectManager");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }



    //Creates instance and makes sure it only creates once
    public static synchronized AutoConnectManager getInstance(Context context) {
        if (instance == null) {
            instance = new AutoConnectManager(context);

        }
        return instance;
    }

    public static synchronized boolean isReady() {
        return instance != null;
    }

    //Starts auto connect
    public void start() {
        if (!isRunning) { 
            Log.i(TAG, "Starting AutoConnect");
            isRunning = true;
            multicastReceiver.startListening(context);//Starts multicast receiver
            handler.post(checkConnectionRunnable);//starts 

        }
    }

    public void stop() {
        if (isRunning) {
            Log.i(TAG, "Stopping AutoConnectManager");
            isRunning = false;
            
            // Stop multicast discovery
            multicastReceiver.stopListening();
            
            // Ta bort pending callbacks
            handler.removeCallbacks(checkConnectionRunnable);
        }
    }

    public void setManualConnection(String ip, int port) {
        Log.i(TAG,"Setting manual connection (highest priority): "  + ip + ":" +  port);
        this.manualIp = ip;
        this.manualPort = port;
        this.isManualOverride = true;
        

        handler.removeCallbacks(checkConnectionRunnable);
        handler.post(checkConnectionRunnable);
    }

    private void clearManualConnection() {
        Log.i(TAG, "Clearing Manual Connection");
        this.manualIp = null;
        this.manualPort = null;
        this.isManualOverride = false;


        handler.removeCallbacks(checkConnectionRunnable);
        handler.post(checkConnectionRunnable);
    }




    //Checks if connected depending on connection-state
    private void checkAndConnect() {
        Log.i(TAG, "Running connection check");
        URI targetUri = findTargetUri();

        if (targetUri == null) {
            Log.i(TAG, "No URI found, trying in" + (CHECK_INTERVAL_MS));
            return;

        }

        WebsocketClientHandler handler = WebsocketClientHandler.getInstance();

        if (handler == null) {
            Log.i(TAG, "No handler exists, creating new" + targetUri);
            createAndConnect(targetUri);
        } else {
            URI handlerUri = handler.getUri();
            if (handlerUri.equals(targetUri)) {
                Log.i(TAG, "Same URI detected, reuse handler");
                if (handler.isConnected()) {
                    Log.i(TAG, "Already connected, maintain connection");
                    return ;
                
                } else if (connecting) {
                    Log.i(TAG, "Connecting, please wait");
                    return ;
                } else {
                    Log.i(TAG, "Not connected, attempting to connect");
                    attemptConnect(handler);
                }
            } else {
                Log.i(TAG,"URI changed from" +  handlerUri + "to" + targetUri + ", resetting handler");
                resetAndConnect(targetUri);
            }
        }
    }




    //Prioritizing to find and create URI
    private URI findTargetUri() {

        //Prioritizes manual input, makes a URI, Check if URI syntax is ok
        if (isManualOverride && manualIp != null && manualPort != null) {
            try {
                URI uri= new URI("ws://" + manualIp + ":" + manualPort);
                Log.i(TAG, "Using manual input" + uri);
                return uri;
            } catch(URISyntaxException e) {
                Log.e(TAG, "Invalid URI", e);
            }
        } 


        //If backend list isn't empty we will take the first ip and port and make a URI
        ArrayList<MulticastReceiver.BackendInformation> backends = multicastReceiver.getAvailableBackends();
        if (!backends.isEmpty())    {
            MulticastReceiver.BackendInformation backend = backends.get(0);
            try {
            URI uri = new URI("ws://" +  backend.ip +":" + backend.port);
            Log.i(TAG, "Using discovered URI via multicast" + uri);
            saveLastConnection(backend.ip, backend.port);
            return uri;
            } catch (URISyntaxException e) {
            Log.e(TAG, "Invalid discovered URI");
        }
        } 
        

        //save URI from last use
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        String lastIp = prefs.getString(PREF_LAST_IP, null);

        int lastPort = prefs.getInt(PREF_LAST_PORT, -1);

        if (lastIp != null && lastPort > 0) {
            try {
                URI uri = new URI("ws://"+ lastIp + ":" + lastPort);
                Log.d(TAG, "Using last used URI from previous use" + uri);
                return uri;

            } catch (URISyntaxException e) {
                Log.e(TAG, "Invalid saved URI", e);
            }
        }
        Log.d(TAG, "No URI found");
        return null; 

    }


    //Creates handler if no handler has been created
    private void createAndConnect(URI uri) {
        connecting = true;
        WebsocketClientHandler newHandler = WebsocketClientHandler.createInstance(context, uri);


        boolean connected = newHandler.connect();

        if (connected) {
            Log.i(TAG, "Connection initiated successfully");
            saveLastConnection(uri);
        } else {
            Log.e(TAG, "Failed to initiate connection");
        }
        connecting = false;
    }

    //if handler but not connected, try to reconnect
    private void attemptConnect(WebsocketClientHandler handler) {
        connecting = true;

        WebsocketClientHandler oldHandler = WebsocketClientHandler.getInstance();
        if (oldHandler != null) {
            oldHandler.closeConnection();
        }

        // Resets and creates handler
        WebsocketClientHandler newHandler = WebsocketClientHandler.resetClientHandler(context, oldHandler.getUri());

        boolean connected = newHandler.connect();
        /*handler.isConnected()

        if (connected) {
            Log.i(TAG, "Reconnection initiated successfully");
        } else {
            Log.i(TAG, "Failed to initiate reconnection");
        }*/
        connecting = false;
    }


    //If new URI found, close-reset-creat-connect to new URI
    private void resetAndConnect(URI newUri) {
        connecting = true;
        
        // Close old connection
        WebsocketClientHandler oldHandler = WebsocketClientHandler.getInstance();
        if (oldHandler != null) {
            oldHandler.closeConnection();
        }
        
        // Resets and creates handler
        WebsocketClientHandler newHandler = WebsocketClientHandler.resetClientHandler(context, newUri);
        
        boolean connected = newHandler.connect();
        
        if (connected) {
            Log.i(TAG, "New connection initiated successfully");
            saveLastConnection(newUri);
        } else {
            Log.e(TAG, "Failed to initiate new connection");
        }
        
        connecting = false;
    }

    private void saveLastConnection(String ip, int port) {
    SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    prefs.edit()
        .putString(PREF_LAST_IP, ip)
        .putInt(PREF_LAST_PORT, port)
        .apply();
    Log.d(TAG,("Saved last connection: " + ip + ":" + port));
    }

    private void saveLastConnection(URI uri) {
        saveLastConnection(uri.getHost(), uri.getPort());
    }
    


    //status
    public boolean isConnected() {
        WebsocketClientHandler handler = WebsocketClientHandler.getInstance();
        return handler != null && handler.isConnected();
    }
    public URI getCurrentUri() {
        WebsocketClientHandler handler = WebsocketClientHandler.getInstance();
        return handler != null ? handler.getUri() : null;
    }
    public void destroy() {
        stop();
        handlerThread.quitSafely();
    }
}