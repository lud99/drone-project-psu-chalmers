package com.dji.sdk.sample;

import android.content.Context;
import android.net.wifi.WifiManager;
import android.util.Log;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.InetAddress;
import java.net.MulticastSocket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.json.JSONObject; // JSON handling
import org.json.JSONException; // JSON error handling

public class MulticastReceiver {
    public static class BackendInformation
    {
        String name;
        String ip;
        int port;

        public BackendInformation(String name, String ip, int port) {
            this.name = name;
            this.ip = ip;
            this.port = port;
        }
    }
    public static final String TAG = MulticastReceiver.class.getName();

    ExecutorService socketExecutor;
    WifiManager wifi;
    WifiManager.MulticastLock multicastLock;

    private final String groupIP = "239.255.42.99";
    private int port = 0;

    private final char[] magicBytes = { 'C', 'T', 'H'};

    private final Lock backendsLock = new ReentrantLock(true);

    public static final int PayloadLength = 1200;

    private final Map<String, BackendInformation> availableBackends = new HashMap<>();

    MulticastReceiver(int port)
    {
        this.port = port;
    }

    public void startListening(Context context)
    {
        wifi = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
        multicastLock = wifi.createMulticastLock("discovery-lock");

        multicastLock.setReferenceCounted(true);
        multicastLock.acquire();

        ExecutorService socketExecutor = Executors.newSingleThreadExecutor();

        socketExecutor.execute(() -> {
            try ( MulticastSocket socket = new MulticastSocket(this.port);) {
                InetAddress group = InetAddress.getByName(groupIP);

                socket.joinGroup(group);

                receivePackets(socket);
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    public void stopListening()
    {
        Log.d(TAG, "Shutting down socket");
        multicastLock.release();
        if (socketExecutor != null) {
            socketExecutor.shutdownNow();
        }
    }

    private void receivePackets(MulticastSocket socket) throws IOException {
        while (!Thread.currentThread().isInterrupted()) {
            byte[] buf = new byte[PayloadLength];
            DatagramPacket packet = new DatagramPacket(buf, buf.length);

            socket.receive(packet);

            handlePacket(packet, buf);
        }

        socket.close();
    }

    public boolean handlePacket(DatagramPacket packet, byte[] buf)
    {
        // Check magic bytes so we have received what we expect
        boolean valid = true;
        for (int i = 0; i < magicBytes.length; i++) {
            if (buf[i] != magicBytes[i]) {
                Log.d(TAG, "Received unexpected multicast packet, ignoring it");
                valid = false;
            }
        }
        if (!valid)
        {
            return false;
        }

        String message = new String(buf, StandardCharsets.UTF_8).substring(3);
        try {
            Log.d(TAG, "Received multicast packet " + message);

            JSONObject jsonMessage = new JSONObject(message);
            String type = jsonMessage.getString("msg_type");
            if (!type.equals("backend_discovery")) {
                Log.d(TAG, "Unknown msg_type: " + type);
                return false;
            }

            String name = jsonMessage.getString("name");
            String ip = jsonMessage.getString("ip");
            int port = jsonMessage.getInt("port");

            backendsLock.lock();
            availableBackends.put(name, new BackendInformation(name, ip, port));
            backendsLock.unlock();

            Log.d(TAG, "Saved backend information");

            return true;
        }
        catch (JSONException e)
        {
            Log.d(TAG, "Failed to parse json " + message);
            e.printStackTrace();
            return false;
        }
    }

    public ArrayList<BackendInformation> getAvailableBackends()
    {
        backendsLock.lock();
        ArrayList<BackendInformation> list = new ArrayList<>(availableBackends.values());
        backendsLock.unlock();

        return list;
    }
}
