package com.dji.sdk.sample;

import android.util.Log;

import org.json.JSONArray;
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
            message.put("msg_type", "taskComplete");
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
            message.put("msg_type", "taskAborted");
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
            message.put("msg_type", "allTasksAborted");
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
            message.put("msg_type", "taskFailed");
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
        WebsocketClientHandler websocketClientHandler = WebsocketClientHandler.getInstance();
        if (websocketClientHandler != null) {
            websocketClientHandler.send(telemetryToJson(telemetry).toString());
        }
    }

    public void registrationData(DroneAdapter.RegistrationData registrationData, DroneAdapter.Telemetry telemetry) {
        WebsocketClientHandler websocketClientHandler = WebsocketClientHandler.getInstance();
        if (websocketClientHandler != null) {
            websocketClientHandler.send(registrationDataToJson(registrationData, telemetry).toString());
        }
    }

    private JSONObject telemetryToJson(DroneAdapter.Telemetry telemetry) {
        JSONObject telemetryJson = new JSONObject();
        if (telemetry == null) {
            return telemetryJson;
        }

        try {
            telemetryJson.put("msg_type", "telemetry");
            telemetryJson.put("drone_id", telemetry.droneID);
            telemetryJson.put("lat", telemetry.lat);
            telemetryJson.put("lon", telemetry.lon);
            telemetryJson.put("alt", telemetry.alt);
            telemetryJson.put("heading", telemetry.heading);
            telemetryJson.put("speed", telemetry.speed);
            telemetryJson.put("battery_percent", telemetry.batteryPercent);
        } catch (Exception e) {
            Log.e("MessageHandler", "Error serializing telemetry:", e);
        }

        return telemetryJson;
    }

    private JSONObject registrationDataToJson(DroneAdapter.RegistrationData registrationData, DroneAdapter.Telemetry telemetry) {
        JSONObject registrationDataJson = new JSONObject();
        if (registrationData == null) {
            return registrationDataJson;
        }

        try {
            registrationDataJson.put("msg_type", "drone_registration");
            registrationDataJson.put("drone_type", registrationData.droneType);
            registrationDataJson.put("model", registrationData.model);
            registrationDataJson.put("drone_id", registrationData.droneID);
            registrationDataJson.put("capabilities", capabilitiesToJson(registrationData.capabilities));
            registrationDataJson.put("telemetry", telemetryDataToJson(telemetry));
        } catch (Exception e) {
            Log.e("MessageHandler", "Error serializing registration data:", e);
        }

        return registrationDataJson;
    }

    private JSONObject telemetryDataToJson(DroneAdapter.Telemetry telemetry) {
        JSONObject telemetryJson = new JSONObject();
        try {
            if (telemetry == null) {
                telemetryJson.put("lat", 0.0);
                telemetryJson.put("lon", 0.0);
                telemetryJson.put("alt", 0.0);
                telemetryJson.put("heading", 0);
                telemetryJson.put("speed", 0.0);
                telemetryJson.put("battery_percent", -1);
            } else {
                telemetryJson.put("lat", telemetry.lat);
                telemetryJson.put("lon", telemetry.lon);
                telemetryJson.put("alt", telemetry.alt);
                telemetryJson.put("heading", telemetry.heading);
                telemetryJson.put("speed", telemetry.speed);
                telemetryJson.put("battery_percent", telemetry.batteryPercent);
            }
        } catch (Exception e) {
            Log.e("MessageHandler", "Error serializing registration telemetry:", e);
        }

        return telemetryJson;
    }

    private JSONObject capabilitiesToJson(DroneAdapter.Capabilities capabilities) {
        JSONObject capabilitiesJson = new JSONObject();
        if (capabilities == null) {
            return capabilitiesJson;
        }

        try {
            capabilitiesJson.put("camera", cameraToJson(capabilities.camera));
            capabilitiesJson.put("led", ledToJson(capabilities.led));
            capabilitiesJson.put("spotlight", capabilities.spotlight);
            if (capabilities.speaker == null) {
                capabilitiesJson.put("speaker", JSONObject.NULL);
            } else {
                capabilitiesJson.put("speaker", speakerToJson(capabilities.speaker));
            }
        } catch (Exception e) {
            Log.e("MessageHandler", "Error serializing capabilities:", e);
        }

        return capabilitiesJson;
    }

    private JSONObject cameraToJson(DroneAdapter.Capabilities.Camera camera) {
        JSONObject cameraJson = new JSONObject();
        if (camera == null) {
            return cameraJson;
        }

        try {
            cameraJson.put("aspect_ratio", camera.aspect_ratio);
            cameraJson.put("horizontal_fov", camera.horizontal_fov);
            cameraJson.put("resolution_height", camera.resolution_height);
            cameraJson.put("resolution_width", camera.resolution_width);
        } catch (Exception e) {
            Log.e("MessageHandler", "Error serializing camera capabilities:", e);
        }

        return cameraJson;
    }

    private JSONObject ledToJson(DroneAdapter.Capabilities.Led led) {
        JSONObject ledJson = new JSONObject();
        if (led == null) {
            return ledJson;
        }

        try {
            ledJson.put("types", stringArrayToJsonArray(led.types));
        } catch (Exception e) {
            Log.e("MessageHandler", "Error serializing LED capabilities:", e);
        }

        return ledJson;
    }

    private JSONObject speakerToJson(DroneAdapter.Capabilities.Speaker speaker) {
        JSONObject speakerJson = new JSONObject();
        if (speaker == null) {
            return speakerJson;
        }

        try {
            speakerJson.put("audio_files", stringArrayToJsonArray(speaker.audio_files));
        } catch (Exception e) {
            Log.e("MessageHandler", "Error serializing speaker capabilities:", e);
        }

        return speakerJson;
    }

    private JSONArray stringArrayToJsonArray(String[] values) {
        JSONArray jsonArray = new JSONArray();
        if (values == null) {
            return jsonArray;
        }

        for (String value : values) {
            jsonArray.put(value);
        }

        return jsonArray;
    }
}