package com.dji.sdk.sample;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;


/*
* Responsible for running test tasks on the drone.
* It provides methods to execute various tasks such as moving to a location, angling the camera, playing audio, and more.
* To add a new test task, simply add a new enum value to TestTask and implement the corresponding method to execute the desired drone action.
*/
public class TaskTestRunner {
    private static final String TAG = TaskTestRunner.class.getSimpleName();

    public enum TestTask {
        GO_TO,
        ANGLE_CAMERA,
        PLAY_AUDIO,
        GO_HOME,
        LED,
        DEACTIVATE_LED,
        SPOTLIGHT,
        ABORT_TASK,
        LAND,
        SEND_TELEMETRY,
        SEND_REGISTRATION_DATA
    }

    private static final TaskTestRunner INSTANCE = new TaskTestRunner();

    private DroneAdapter adapter;
    private TestTask activeTask = TestTask.GO_HOME;
    private int taskCounter = 1;

    private TaskTestRunner() {
        adapter = DroneAdapterManager.getCurrentAdapter();
    }

    public static TaskTestRunner getInstance() {
        return INSTANCE;
    }

    public void setAdapter(DroneAdapter adapter) {
        if (adapter != null) {
            this.adapter = adapter;
        }
    }

    public void setActiveTask(TestTask task) {
        if (task != null) {
            activeTask = task;
        }
    }

    public TestTask getActiveTask() {
        return activeTask;
    }

    public void runActiveTask() {
        adapter = DroneAdapterManager.getCurrentAdapter();

        if (adapter == null) {
            Log.e(TAG, "No adapter configured for task tests.");
            return;
        }

        String missionId = "test-mission";
        int taskIndex = taskCounter++;

        switch (activeTask) {
            case GO_TO:
                runGoTo(missionId, taskIndex);
                break;
            case ANGLE_CAMERA:
                runAngleCamera(missionId, taskIndex);
                break;
            case PLAY_AUDIO:
                runPlayAudio(missionId, taskIndex);
                break;
            case GO_HOME:
                runGoHome(missionId, taskIndex);
                break;
            case LED:
                runLed(missionId, taskIndex);
                break;
            case DEACTIVATE_LED:
                runDeactivateLed(missionId, taskIndex);
                break;
            case SPOTLIGHT:
                runSpotlight(missionId, taskIndex);
                break;
            case ABORT_TASK:
                runAbortTask(missionId, taskIndex);
                break;
            case LAND:
                runLand(missionId, taskIndex);
                break;
            case SEND_TELEMETRY:
                runSendTelemetry(missionId, taskIndex);
                break;
            case SEND_REGISTRATION_DATA:
                runSendRegistrationData(missionId, taskIndex);
                break;
            default:
                Log.w(TAG, "Unknown active task: " + activeTask);
        }
    }

    public void runGoTo(String missionId, int taskIndex) {
        DroneAdapter.Telemetry telemetry = adapter.getTelemetry();
        if (telemetry == null) {
            Log.e(TAG, "Telemetry unavailable; GO_TO test skipped.");
            return;
        }

        // double testLat = telemetry.lat + 0.0003;
        // double testLon = telemetry.lon + 0.0003;
        // float testAlt = 30.0f;
        // int heading = 90;
        double testLat = 57.695587;
        double testLon = 11.991797;
        float testAlt = 3.0f;
        Integer heading = null;
        adapter.goTo(testLat, testLon, testAlt, heading, missionId, taskIndex);
    }

    public void runAngleCamera(String missionId, int taskIndex) {
        adapter.angleCamera(-20.0f, 30.0f, 10.0f, missionId, taskIndex);
    }

    public void runPlayAudio(String missionId, int taskIndex) {
        adapter.playAudio("instructions", 0.5f, 5, missionId, taskIndex);
    }

    public void runGoHome(String missionId, int taskIndex) {
        adapter.goHome(missionId, taskIndex);
    }

    public void runLed(String missionId, int taskIndex) {
        adapter.led("beacon", 5, missionId, taskIndex);
    }

    public void runSpotlight(String missionId, int taskIndex) {
        adapter.spotlight(0.5f, null, missionId, taskIndex);
    }

    public void runDeactivateLed(String missionId, int taskIndex) {
        adapter.deactivateLed("front", missionId, taskIndex);
    }

    public void runAbortTask(String missionId, int taskIndex) {
        adapter.abortTask(missionId, taskIndex, "go_to");
    }

    public void runLand(String missionId, int taskIndex) {
        adapter.land(missionId, taskIndex);
    }

    public void runSendTelemetry(String missionId, int taskIndex) {
        DroneAdapter.Telemetry telemetry = adapter.getTelemetry();
        if (telemetry == null) {
            Log.e(TAG, "Telemetry unavailable; SEND_TELEMETRY skipped.");
            return;
        }

        try {
            JSONObject message = new JSONObject();
            message.put("msg_type", "telemetry");
            message.put("mission_id", missionId);
            message.put("task_index", taskIndex);
            message.put("droneID", telemetry.droneID == null ? "unknown" : telemetry.droneID);
            message.put("lat", telemetry.lat);
            message.put("lon", telemetry.lon);
            message.put("alt", telemetry.alt);
            message.put("heading", telemetry.heading);
            message.put("speed", telemetry.speed);
            message.put("batteryPercent", telemetry.batteryPercent);

            Log.i(TAG, "Telemetry test payload: " + message.toString());
        } catch (JSONException e) {
            Log.e(TAG, "Failed to build telemetry test message", e);
        }
    }

    public void runSendRegistrationData(String missionId, int taskIndex) {
        adapter.getRegistrationData(new DroneAdapter.RegistrationDataCallback() {
            @Override
            public void onSuccess(DroneAdapter.RegistrationData registrationData) {
                logRegistrationDataPayload(missionId, taskIndex, registrationData);
            }

            @Override
            public void onFailure(String reason) {
                Log.e(TAG, "Registration data unavailable; SEND_REGISTRATION_DATA skipped. Reason: " + reason);
            }
        });
    }

    private void logRegistrationDataPayload(String missionId, int taskIndex, DroneAdapter.RegistrationData registrationData) {
        if (registrationData == null) {
            Log.e(TAG, "Registration data unavailable; SEND_REGISTRATION_DATA skipped.");
            return;
        }

        try {
            JSONObject message = new JSONObject();
            message.put("msg_type", "registration_data");
            message.put("mission_id", missionId);
            message.put("task_index", taskIndex);
            message.put("droneType", registrationData.droneType == null ? "unknown" : registrationData.droneType);
            message.put("model", registrationData.model == null ? "unknown" : registrationData.model);
            message.put("droneID", registrationData.droneID == null ? "unknown" : registrationData.droneID);

            JSONObject capabilities = new JSONObject();
            DroneAdapter.Capabilities adapterCapabilities = registrationData.capabilities;
            if (adapterCapabilities != null) {
                capabilities.put("spotlight", adapterCapabilities.spotlight);

                if (adapterCapabilities.camera != null) {
                    JSONObject camera = new JSONObject();
                    camera.put("aspect_ratio", adapterCapabilities.camera.aspect_ratio);
                    camera.put("diagonal_fov", adapterCapabilities.camera.diagonal_fov);
                    camera.put("resolution_height", adapterCapabilities.camera.resolution_height);
                    camera.put("resolution_width", adapterCapabilities.camera.resolution_width);
                    capabilities.put("camera", camera);
                }

                if (adapterCapabilities.led != null && adapterCapabilities.led.types != null) {
                    JSONArray ledTypes = new JSONArray();
                    for (String type : adapterCapabilities.led.types) {
                        ledTypes.put(type);
                    }
                    capabilities.put("led_types", ledTypes);
                }

                if (adapterCapabilities.speaker != null && adapterCapabilities.speaker.audio_files != null) {
                    JSONArray audioFiles = new JSONArray();
                    for (String file : adapterCapabilities.speaker.audio_files) {
                        audioFiles.put(file);
                    }
                    capabilities.put("audio_files", audioFiles);
                }
            }

            message.put("capabilities", capabilities);
            Log.i(TAG, "RegistrationData test payload: " + message.toString());
        } catch (JSONException e) {
            Log.e(TAG, "Failed to build registration test message", e);
        }
    }
}
