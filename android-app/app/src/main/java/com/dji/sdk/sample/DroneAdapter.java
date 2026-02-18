/**
     * Contains the structure for the adapter for each drone type.
     * Each drone type should have it own class "<drone_type>Adapter" Eg. DJIAdapter/MavlinkAdapter that follows this structure.
     * Contains both methods to execute tasks and retrieve data
     * The methods that executes tasks should also include the functionality that sends "task_complete"-messages to backend
*/


public interface DroneAdapter {

    public static Telemetry {
        String droneID;
        double lat;
        double long;
        float alt;
        int heading;
        float speed;
        int batteryPercent;
    }

    public static RegistrationData {
        String droneType;
        String model;
        String droneID;
        Capabilities capabilities;
    }

    public static class Capabilities {
        public Camera camera = null;
        public Led led = null;
        public boolean spotlight = false;
        public boolean speaker = false;
        public int maxSpeed = null;

        public static class Camera {
            public String aspect_ratio = null;
            public double horizontal_fov = null;
            public int resolution = null;
        }

        public static class Led {
            public String[] colors = {}; // Will likely have to changed
        }
    }

    void goTo(double lat, double lon, float alt, Integer heading, String missionID, int taskIndex);
    void angleCamera(float pitch, float yaw, float transitionTime, String missionID, int taskIndex);
    void playAudio(String file, float volume, Integer durationSeconds, String missionID, int taskIndex);
    void goHome(int taskIndex);
    void led(String color, String pattern, Integer durationSeconds, String missionID, int taskIndex)
    void spotlight(String pattern, int durationSecond, String missionID, int taskIndex)

    void abortTask(String missionID, int taskIndex);
    void land(String missionID, int taskIndex);

    Telemetry getTelemetry(); 
    RegistrationData getRegistrationData();
}