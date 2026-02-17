package com.dji.sdk.sample;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import android.view.TextureView;
import android.content.Context;


import com.dji.sdk.sample.databinding.ActivityServerBinding;

import org.w3c.dom.Text;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Locale;

import dev.gustavoavila.websocketclient.WebSocketClient;
import dji.thirdparty.afinal.core.AsyncTask;

public class ServerActivity extends AppCompatActivity {
    private final String TAG = ServerActivity.class.getName();
    private WebsocketClientHandler websocketClientHandler;
    EditText ipTextEdit;
    EditText portEdit;



    private AutoConnectManager autoConnectManager;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_server);
        ipTextEdit = findViewById(R.id.ip_adress_edit);
        portEdit = findViewById(R.id.portEdit);
        
        //
        autoConnectManager = AutoConnectManager.getInstance(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        //autoConnectManager.start();
        updateStatus();  //Start updating the status as soon as the activity is resumed.
    }
    
    @Override
    protected void onPause() {
        super.onPause();
    }
    /**
     * Handles the onclick event of the connect button. First, it creates a URI based on
     * the IP and Port fields. Then, it works what webSocketClientHandler to use and stores that.
     * Finally, it connects if the current client isn't already connected.
     */
    


    public void connectClick(View v) {
        Log.e(TAG, "button clicked!");
        URI newUri;
        try {
            newUri = new URI("ws://" + ipTextEdit.getText() + ":" + portEdit.getText());
        } catch (URISyntaxException e) {
            Toast.makeText(this, "Incorrectly formatted URI", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            String ip = ipTextEdit.getText().toString();
            int port = Integer.parseInt(portEdit.getText().toString());
            autoConnectManager.setManualConnection(ip, port);
            Log.i(TAG, "Manual connection set: " + ip + ":" + port);
        } catch (NumberFormatException e) {
            Log.e(TAG, "Invalid port number", e);
        }
        toastOnUIThread("Manual connection set. Connecting...");
    }


    /**
     * Sends a simple websocket message, for debugging.
     */
    public void sendClick(View v) {
        Log.e(TAG, "send clicked!");
    
    // LÄGG TILL DESSA RADER:
        WebsocketClientHandler handler = WebsocketClientHandler.getInstance();
        if (handler == null || !handler.isConnected()) {
            Toast.makeText(this, "Not connected", Toast.LENGTH_SHORT).show();
            return;
        }
    
        String message = "{\"msg_type\": \"Debug\",\"msg\": \"Hello, from Android!\"}";
        handler.send(message);  // ÄNDRAT: använd handler istället för websocketClientHandler
    }

    /**
     * A simpler toast method that can be called from Async
     * @param message The message to toast
     */
    private void toastOnUIThread(String message) {
        runOnUiThread(() -> Toast.makeText(ServerActivity.this, message, Toast.LENGTH_SHORT).show());
    }

    /**
     * Continually update the status.
     * Currently only checking for Connection/No connection/No instance
     */
    private void updateStatus() {
        AsyncTask.execute(() -> {
            //This uses busy-wait, which isn't great...
            while (true) {
                try {
                    runOnUiThread(updateRunnable);
                    WebsocketClientHandler.status_update.acquire();
                } catch (InterruptedException e) {
                    Log.e(TAG, "interrupted!");
                }
            }
        });
    }

    private final Runnable updateRunnable = new Runnable() {
    @Override
    public void run() {
        TextView connectionStatusView = findViewById(R.id.banankaka);
        
        // LÄGG TILL DENNA RAD:
        WebsocketClientHandler handler = WebsocketClientHandler.getInstance();
        
        // ÄNDRA FRÅN websocketClientHandler TILL handler:
        if (handler != null && handler.isConnected()) {
            connectionStatusView.setText(String.format("Connected to: %s", handler.getUri()));
        } else if (handler != null) {
            connectionStatusView.setText(String.format("Disconnected from: %s", handler.getUri()));
        } else {
            connectionStatusView.setText("No Client initialized");
        }
        }
    };
}
