package com.dji.sdk.sample;

/**
     * Contains the structure for the adapter for each drone type.
     * Each drone type should have it own class "<drone_type>Adapter" Eg. DJIAdapter/MavlinkAdapter that follows this structure.
     * Contains both methods to execute tasks and retrieve data
     * The methods that executes tasks should also include the functionality that sends "task_complete"-messages to backend
*/


public interface DroneAdapter {

    class Telemetry {
        String droneID;
        double lat;
        double lon;
        float alt;
        int heading;
        float speed;
        int batteryPercent;
    }

    static class RegistrationData {
        String droneType;
        String model;
        String droneID;
        Capabilities capabilities;
    }

    static class Capabilities {
        public Camera camera = null;
        public Led led = null;
        public boolean spotlight = false;
        public boolean speaker = false;
        public Integer maxSpeed = null;

        public static class Camera {
            public Float aspect_ratio = null;
            public Float horizontal_fov = null;
            public Integer resolution_height = null;
            public Integer resolution_width = null;
        }

        public static class Led {
            public String[] colors = {}; // Will likely have to changed
        }
    }

    void goTo(double lat, double lon, float alt, Integer heading, String missionID, int taskIndex);
    void angleCamera(float pitch, float yaw, Float transitionTime, String missionID, int taskIndex);
    void playAudio(String file, float volume, Integer durationSeconds, String missionID, int taskIndex);
    void goHome(String missionID, int taskIndex);
    void led(String color, String pattern, Integer durationSeconds, String missionID, int taskIndex);
    void spotlight(float brightness, Integer durationSeconds, String missionID, int taskIndex);

    void abortTask(String missionID, int taskIndex);
    void land(String missionID, int taskIndex);

    Telemetry getTelemetry(); 
    RegistrationData getRegistrationData();
    
    void pushTaskComplete(String missionID, int taskIndex);
    void pushTaskFailed(String missionID, int taskIndex);
}