package com.dji.sdk.sample;

import android.util.Log;

import org.json.JSONObject;




class MessageHandler {
    private static MessageHandler instance;

    private MessageHandler() {
    }

    public static MessageHandler getInstance() {
        if (instance == null) {
            instance = new MessageHandler();
        }
        return instance;
    }

    public void taskComplete(String currentMissionID, int currentTaskIndex) {
        JSONObject message = new JSONObject();
        try {
            message.put("type", "taskComplete");
            message.put("missionID", currentMissionID);
            message.put("taskIndex", currentTaskIndex);
        } catch(Exception e) {
            Log.e("MessageHandler", "Error creating JSON message:", e);
        }
        WebsocketClientHandler websocketClientHandler = WebsocketClientHandler.getInstance();
        if (websocketClientHandler != null) {
            websocketClientHandler.send(message.toString());
        }
    }

    public void taskAborted(String currentMissionID, int taskIndex, String taskType) {
        JSONObject message = new JSONObject();
        try {
            message.put("type", "taskAborted");
            message.put("missionID", currentMissionID);
            message.put("taskIndex", taskIndex);
            message.put("taskType", taskType);
        } catch (Exception e) {
            Log.e("MessageHandler", "Error creating JSON message", e);
        }
        WebsocketClientHandler websocketClientHandler = WebsocketClientHandler.getInstance();
        if (websocketClientHandler != null) {
            websocketClientHandler.send(message.toString());
        }
    }

    public void allTasksAborted(String currentMissionID) {
        JSONObject message = new JSONObject();
        try {
            message.put("type", "allTasksAborted");
            message.put("missionID", currentMissionID);
        } catch(Exception e) {
            Log.e("MessageHandler", "Error creating JSON message:", e);
        }
        WebsocketClientHandler websocketClientHandler = WebsocketClientHandler.getInstance();
        if (websocketClientHandler != null) {
            websocketClientHandler.send(message.toString());
        }
    }

    public void taskFailed(String currentMissionID, int currentTaskIndex) {
        JSONObject message = new JSONObject();
        try {
            message.put("type", "taskFailed");
            message.put("missionID", currentMissionID);
            message.put("taskIndex", currentTaskIndex);
        } catch(Exception e) {
            Log.e("MessageHandler", "Error creating JSON message:", e);
        }
        WebsocketClientHandler websocketClientHandler = WebsocketClientHandler.getInstance();
        if (websocketClientHandler != null) {
            websocketClientHandler.send(message.toString());
        }
    }

    public void telemetryUpdate(DroneAdapter.Telemetry telemetry) {
        JSONObject message = new JSONObject();
        try {
            message.put("type", "telemetryUpdate");
            message.put("telemetry", telemetry);
        } catch(Exception e) {
            Log.e("MessageHandler", "Error creating JSON message:", e);
        }
        WebsocketClientHandler websocketClientHandler = WebsocketClientHandler.getInstance();
        if (websocketClientHandler != null) {
            websocketClientHandler.send(message.toString());
        }
    }

    public void registrationData(DroneAdapter.RegistrationData registrationData) {
        JSONObject message = new JSONObject();
        try {
            message.put("type", "registrationData");
            message.put("registrationData", registrationData);
        } catch(Exception e) {
            Log.e("MessageHandler", "Error creating JSON message:", e);
        }
        WebsocketClientHandler websocketClientHandler = WebsocketClientHandler.getInstance();
        if (websocketClientHandler != null) {
            websocketClientHandler.send(message.toString());
        }
    }
}