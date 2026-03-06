package com.dji.sdk.sample;

import static dji.midware.data.manager.P3.ServiceManager.getContext;

import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import dji.common.camera.ResolutionAndFrameRate;
import dji.common.camera.SettingsDefinitions;
import dji.common.error.DJIError;
import dji.common.flightcontroller.FlightControllerState;
import dji.common.flightcontroller.GPSSignalLevel;
import dji.common.gimbal.Rotation;
import dji.common.gimbal.RotationMode;
import dji.common.mission.waypoint.Waypoint;
import dji.common.mission.waypoint.WaypointAction;
import dji.common.mission.waypoint.WaypointActionType;
import dji.common.mission.waypoint.WaypointMission;
import dji.common.mission.waypoint.WaypointMissionDownloadEvent;
import dji.common.mission.waypoint.WaypointMissionExecutionEvent;
import dji.common.mission.waypoint.WaypointMissionFinishedAction;
import dji.common.mission.waypoint.WaypointMissionFlightPathMode;
import dji.common.mission.waypoint.WaypointMissionHeadingMode;
import dji.common.mission.waypoint.WaypointMissionUploadEvent;
import dji.common.battery.BatteryState;
import dji.common.util.CommonCallbacks;
import dji.sdk.flightcontroller.FlightController;
import dji.sdk.mission.waypoint.WaypointMissionOperator;
import dji.sdk.mission.waypoint.WaypointMissionOperatorListener;
import dji.sdk.products.Aircraft;
import dji.sdk.sdkmanager.DJISDKManager;
import android.util.ArrayMap;
import dji.common.flightcontroller.LEDsSettings;
import dji.sdk.accessory.speaker.Speaker;
import dji.sdk.accessory.spotlight.Spotlight;
import dji.common.gimbal.Attitude;
import dji.common.accessory.SettingsDefinitions.PlayMode;
import dji.sdk.base.BaseProduct;
import dji.common.mission.waypoint.WaypointMissionState;





class DJIFlightManager {
    private final List<Waypoint> waypointList = new ArrayList<>(); //List for storing waypoint / -s
    private final List<WaypointAction> primarywaypointActions = new ArrayList<>();
    public static WaypointMission.Builder waypointMissionBuilder;
    private WaypointMissionOperator waypointMissionOperator;
    private WaypointMissionFinishedAction mFinishedAction = WaypointMissionFinishedAction.NO_ACTION;
    private WaypointMissionHeadingMode mHeadingMode = WaypointMissionHeadingMode.AUTO;
    private WaypointMissionFlightPathMode mFlightPathMode = WaypointMissionFlightPathMode.NORMAL;
    private FlightController controller;
    private BatteryState batteryState;
    private static DJIFlightManager flightManager = null;
    private FlightControllerState state;
    private Aircraft aircraft;
    private CoordinatesActivity coordinatesActivity;
    public double input_lat;
    public double input_lng;
    public float input_alt;
    public int input_yaw;
    private WaypointMissionOperatorListener eventNotificationListener;
    private String currentMissionID;
    private int currentTaskIndex;
    private String cachedSerialNumber = null;
    private String cachedModel = null;

    private Map<String, Integer> audioIndexCache = new ArrayMap<>();

    private Attitude currentGimbalAttitude;


    private DJIFlightManager(){
        this.aircraft = (Aircraft) DJISDKManager.getInstance().getProduct();
        aircraft.getFlightController().setStateCallback(new FlightControllerState.Callback() {
            @Override
            public void onUpdate(@NonNull FlightControllerState flightControllerState) {
                state = flightControllerState;
            }
        });
        aircraft.getBattery().setStateCallback(new BatteryState.Callback() {
            @Override
            public void onUpdate(BatteryState batteryState) {
                DJIFlightManager.this.batteryState = batteryState;
            }
        });
        controller = aircraft.getFlightController();

        if (aircraft.getGimbal() != null) {
            aircraft.getGimbal().setStateCallback(new dji.common.gimbal.GimbalState.Callback() {
                @Override
                public void onUpdate(@NonNull dji.common.gimbal.GimbalState gimbalState) {
                    // Saves pitch, roll, yaw every time it changes
                    currentGimbalAttitude = gimbalState.getAttitudeInDegrees();
                }
            });
        }

        /**
         * See DJI Google Maps demo.
         */
        eventNotificationListener = new WaypointMissionOperatorListener() {
            @Override
            public void onDownloadUpdate(WaypointMissionDownloadEvent downloadEvent) {
            }
            @Override
            public void onUploadUpdate(WaypointMissionUploadEvent uploadEvent) {
            }
            @Override
            public void onExecutionUpdate(WaypointMissionExecutionEvent executionEvent) {
            }
            @Override
            public void onExecutionStart() {
            }
            @Override
            public void onExecutionFinish(@Nullable final DJIError error) {
                if (error == null) {
                   
                    Log.d("DJI", "Waypoint reached for task: " + currentTaskIndex);
                    
                    if (MessageHandler.getInstance() != null) {
                        MessageHandler.getInstance().taskComplete(currentMissionID, currentTaskIndex);
                    }
                } else {
                    // Elso add message to backend here
                    Log.e("DJI", "Mission failed: " + error.getDescription());
                    if (MessageHandler.getInstance() != null) {
                        MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex);
                    }
                }
            }
        };
    }

    /**
     * Returns the singleton instance of the FlightManager class. If the FlightManger is not
     * instantiated, it is.
     * @return The singleton instance of FlightManager
     */
    public static synchronized DJIFlightManager getFlightManager(){
        if (flightManager == null){
            flightManager = new DJIFlightManager();
        }
        return flightManager;
    }

    /**
     * @return The WaypointMissionOperator if the drone is connected, otherwise null
     */
    @Nullable
    public WaypointMissionOperator getWaypointMissionOperator() {
        if (waypointMissionOperator == null) {
            if (DJISDKManager.getInstance().getMissionControl() != null){
                waypointMissionOperator = DJISDKManager.getInstance().getMissionControl().getWaypointMissionOperator();
            }
        }
        return waypointMissionOperator;
    }

    public FlightController getController() {
        return controller;
    }

    public FlightControllerState getState() {
        return state;
    }

    public Aircraft getAircraft() {
        return aircraft;
    }
    public BatteryState getBatteryState(){
        return batteryState;
    }

    public DroneAdapter.Telemetry getTelemetry() {
        if (state == null) {
            return null;
        }

        dji.common.flightcontroller.LocationCoordinate3D location = state.getAircraftLocation();
        if (location == null) {
            return null;
        }


        double lat = location.getLatitude();
        double lon = location.getLongitude();
        if (Double.isNaN(lat) || Double.isNaN(lon) || (lat == 0.0 && lon == 0.0)) {
            return null;
        }

        DroneAdapter.Telemetry telemetry = new DroneAdapter.Telemetry();
        if (cachedSerialNumber == null) {
            getRegistrationData(); // Attempt to fetch and cache the serial number if not already done
        }
        telemetry.droneID = cachedSerialNumber == null ? "unknown" : cachedSerialNumber;
        telemetry.lat = lat;
        telemetry.lon = lon;
        telemetry.alt = location.getAltitude();
        telemetry.heading = state.getAttitude() == null ? 0 : (int) Math.round(state.getAttitude().yaw);

        float velocityX = state.getVelocityX();
        float velocityY = state.getVelocityY();
        if (!Float.isNaN(velocityX) && !Float.isNaN(velocityY)) {
            telemetry.speed = (float) Math.sqrt(velocityX * velocityX + velocityY * velocityY);
        } else {
            telemetry.speed = Float.NaN;
        }

        telemetry.batteryPercent = batteryState == null ? -1 : batteryState.getChargeRemainingInPercent();

        return telemetry;
    }

    public CoordinatesActivity getCoordinatesActivity() {
        return coordinatesActivity;
    }

    private interface RegistrationStepDone {
        void done();
    }

    public DroneAdapter.RegistrationData getRegistrationData() {
        BaseProduct product = DJISDKManager.getInstance().getProduct();
        if (!(product instanceof Aircraft)) {
            Log.e("DJI", "getRegistrationData misslyckades: Ingen drönare ansluten.");
            return null;
        }

        this.aircraft = (Aircraft) product;
        DroneAdapter.RegistrationData registrationData = createRegistrationDataSnapshot(this.aircraft);

        getRegistrationDataAsync(new CommonCallbacks.CompletionCallbackWith<DroneAdapter.RegistrationData>() {
            @Override
            public void onSuccess(DroneAdapter.RegistrationData result) {
                if (result != null) {
                    Log.d("DJI", "Registration data async refresh completed.");
                }
            }

            @Override
            public void onFailure(DJIError djiError) {
                String reason = djiError == null ? "unknown" : djiError.getDescription();
                Log.w("DJI", "Registration data async refresh failed: " + reason);
            }
        });

        return registrationData;
    }

    public void getRegistrationDataAsync(CommonCallbacks.CompletionCallbackWith<DroneAdapter.RegistrationData> callback) {
        BaseProduct product = DJISDKManager.getInstance().getProduct();
        if (!(product instanceof Aircraft)) {
            Log.e("DJI", "getRegistrationDataAsync misslyckades: Ingen drönare ansluten.");
            if (callback != null) {
                callback.onFailure(DJIError.COMMON_DISCONNECTED);
            }
            return;
        }

        this.aircraft = (Aircraft) product;
        DroneAdapter.RegistrationData registrationData = createRegistrationDataSnapshot(this.aircraft);

        fetchDroneIDAsync(this.aircraft, registrationData, () ->
                fetchCameraCapabilitiesAsync(this.aircraft, registrationData, () ->
                        fetchLedCapabilitiesAsync(this.aircraft, registrationData, () ->
                                fetchSpeakerCapabilitiesAsync(this.aircraft, registrationData, () -> {
                                    Log.d("DJI", "Registration data fetched: Type=" + registrationData.droneType + ", Model=" + registrationData.model + ", DroneID=" + registrationData.droneID);
                                    if (callback != null) {
                                        callback.onSuccess(registrationData);
                                    }
                                })
                        )
                )
        );
    }

    private DroneAdapter.RegistrationData createRegistrationDataSnapshot(Aircraft aircraft) {
        DroneAdapter.RegistrationData registrationData = new DroneAdapter.RegistrationData();
        registrationData.capabilities = new DroneAdapter.Capabilities();

        fetchModel(aircraft, registrationData);
        registrationData.droneID = cachedSerialNumber == null ? "Unknown" : cachedSerialNumber;
        fetchSpotlightCapabilities(aircraft, registrationData);
        fetchSpeakerCapabilities(aircraft, registrationData);

        DroneAdapter.Capabilities.Led ledCapabilities = new DroneAdapter.Capabilities.Led();
        ledCapabilities.types = new String[0];
        registrationData.capabilities.led = ledCapabilities;

        return registrationData;
    }

    private void fetchModel(Aircraft aircraft, DroneAdapter.RegistrationData registrationData) {
        registrationData.droneType = "DJI";
        registrationData.model = aircraft.getModel() != null ? aircraft.getModel().getDisplayName() : (cachedModel == null ? "Unknown" : cachedModel);
        cachedModel = registrationData.model;
    }

    private void fetchDroneIDAsync(Aircraft aircraft, DroneAdapter.RegistrationData registrationData, RegistrationStepDone done) {
        if (aircraft.getFlightController() != null) {
            aircraft.getFlightController().getSerialNumber(new CommonCallbacks.CompletionCallbackWith<String>() {
                @Override
                public void onSuccess(String serialNumber) {
                    if (serialNumber != null && !serialNumber.isEmpty()) {
                        cachedSerialNumber = serialNumber;
                        registrationData.droneID = serialNumber;
                        Log.d("DJI", "Drone ID set to Serial Number: " + serialNumber);
                    } else {
                        registrationData.droneID = "Unknown";
                    }
                    done.done();
                }

                @Override
                public void onFailure(DJIError djiError) {
                    Log.e("DJI", "Failed to fetch Serial Number: " + djiError.getDescription());
                    registrationData.droneID = cachedSerialNumber == null ? "Unknown" : cachedSerialNumber;
                    done.done();
                }
            });
        } else {
            registrationData.droneID = cachedSerialNumber == null ? "Unknown" : cachedSerialNumber;
            done.done();
        }
    }

    private void fetchCameraCapabilitiesAsync(Aircraft aircraft, DroneAdapter.RegistrationData registrationData, RegistrationStepDone done) {
        if (aircraft.getCamera() == null) {
            done.done();
            return;
        }

        dji.sdk.camera.Camera camera = aircraft.getCamera();
        if (camera == null) {
            done.done();
            return;
        }

        DroneAdapter.Capabilities.Camera cameraCapabilities = new DroneAdapter.Capabilities.Camera();
        camera.getVideoResolutionAndFrameRate(new CommonCallbacks.CompletionCallbackWith<ResolutionAndFrameRate>() {
            @Override
            public void onSuccess(ResolutionAndFrameRate settings) {
                if (settings != null) {
                    SettingsDefinitions.VideoResolution djiRes = settings.getResolution();
                    DJIDroneSpecs.Resolution res = DJIDroneSpecs.getDimensions(djiRes);

                    cameraCapabilities.resolution_width = res.width;
                    cameraCapabilities.resolution_height = res.height;

                    SettingsDefinitions.VideoFov djiFov = settings.getFov();
                    cameraCapabilities.horizontal_fov = (float) DJIDroneSpecs.getHorizontalFov(aircraft.getModel(), djiFov);
                    cameraCapabilities.aspect_ratio = (res.height > 0) ? (float) res.width / res.height : 0.0f;

                    registrationData.capabilities.camera = cameraCapabilities;
                    Log.d("DJI", "Camera capabilities updated: " + djiRes.name());
                }
                done.done();
            }

            @Override
            public void onFailure(DJIError djiError) {
                Log.e("DJI", "Failed to get resolution/FOV: " + djiError.getDescription());
                done.done();
            }
        });
    }

    private void fetchLedCapabilitiesAsync(Aircraft aircraft, DroneAdapter.RegistrationData registrationData, RegistrationStepDone done) {
        List<String> ledTypes = new ArrayList<>();
        if (aircraft.getAccessoryAggregation() != null && aircraft.getAccessoryAggregation().getBeacon() != null) {
            ledTypes.add("beacon");
        }

        if (controller == null) {
            DroneAdapter.Capabilities.Led ledCapabilities = new DroneAdapter.Capabilities.Led();
            ledCapabilities.types = ledTypes.toArray(new String[0]);
            registrationData.capabilities.led = ledCapabilities;
            done.done();
            return;
        }

        controller.getLEDsEnabledSettings(new CommonCallbacks.CompletionCallbackWith<LEDsSettings>() {
            @Override
            public void onSuccess(LEDsSettings ledsSettings) {
                if (ledsSettings != null) {
                    // If settings object exists, front/rear LEDs are supported even if currently off.
                    ledTypes.add("front");
                    ledTypes.add("rear");

                    boolean frontOn = ledsSettings.areFrontLEDsOn();
                    boolean rearOn = ledsSettings.areRearLEDsOn();
                    Log.d("DJI", "LED capabilities fetched: " + ledTypes + " (current state front=" + frontOn + ", rear=" + rearOn + ")");
                }

                DroneAdapter.Capabilities.Led ledCapabilities = new DroneAdapter.Capabilities.Led();
                ledCapabilities.types = ledTypes.toArray(new String[0]);
                registrationData.capabilities.led = ledCapabilities;
                done.done();
            }

            @Override
            public void onFailure(DJIError djiError) {
                Log.e("DJI", "Failed to fetch LED settings: " + djiError.getDescription());
                DroneAdapter.Capabilities.Led ledCapabilities = new DroneAdapter.Capabilities.Led();
                ledCapabilities.types = ledTypes.toArray(new String[0]);
                registrationData.capabilities.led = ledCapabilities;
                done.done();
            }
        });
    }

    private void fetchSpotlightCapabilities(Aircraft aircraft, DroneAdapter.RegistrationData registrationData) {
        registrationData.capabilities.spotlight = aircraft.getAccessoryAggregation() != null && aircraft.getAccessoryAggregation().getSpotlight() != null;
    }

    private void fetchSpeakerCapabilities(Aircraft aircraft, DroneAdapter.RegistrationData registrationData) {
        registrationData.capabilities.speaker = AudioFileMapping.buildSpeakerCapabilitiesAndCache(aircraft, audioIndexCache);
    }

    private void fetchSpeakerCapabilitiesAsync(Aircraft aircraft, DroneAdapter.RegistrationData registrationData, RegistrationStepDone done) {
        if (!AudioFileMapping.hasSpeaker(aircraft)) {
            fetchSpeakerCapabilities(aircraft, registrationData);
            done.done();
            return;
        }

        Speaker speaker = aircraft.getAccessoryAggregation().getSpeaker();
        if (speaker == null) {
            fetchSpeakerCapabilities(aircraft, registrationData);
            done.done();
            return;
        }

        speaker.refreshFileList(djiError -> {
            if (djiError != null) {
                Log.w("DJI", "Failed to refresh speaker file list: " + djiError.getDescription());
            }
            fetchSpeakerCapabilities(aircraft, registrationData);
            done.done();
        });
    }


    /**
     * This function starts the process of setting up the waypoint mission and is called when
     * the user presses the Arm button.
     * Firstly we check the batterystate to ensure the drones battery is above 20% as the the
     * application just crashes otherwise.
     * Secondly checking that gps connection is ensured
     * To ensure a correct mission, possible leftovers of previous missions is deleted.
     * It fetches the coordinates of the drone on the ground and creating the first waypoint 10 m above.
     * Thereafter we fetch the test origin coordinates loaded from either the user manually or from ATOS.
     * The waypoints are then put in a list by order of execution,
     * so our primary waypoint is added lastly.
     * Lastly we call configWaypointMission() to further specify the behavior of the waypointMission and
     * uploadWaypointMission() to send the finished product to the drone
     *
     */

    public void GoToWaypoint(double waypoint_lat, double waypoint_lon, float waypoint_alt, Integer waypoint_heading, String missionID, int taskIndex){
        Log.d("DJI", "GoToWaypoint called with lat: " + waypoint_lat + ", lon: " + waypoint_lon + ", alt: " + waypoint_alt + ", heading: " + waypoint_heading);
        // Set these since the Args dont "survive" the entire mission
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        //Checking battery before takeoff to prevent application crash
        int batteryPercent = batteryState.getChargeRemainingInPercent();
        if (batteryPercent <= 20) {
            Toast.makeText(getContext(), "Battery to low to start mission, needs above 20%. Is: "+batteryPercent+"%", Toast.LENGTH_LONG).show();
            return;
        }

        // Checking GPS before start to prevent crashes during next stage (GPS can be 1-5 and NONE)
        GPSSignalLevel gpsSignalLevel = state.getGPSSignalLevel();
        if ((Integer) gpsSignalLevel.value() <= 1 || gpsSignalLevel == GPSSignalLevel.NONE){
            Toast.makeText(getContext(), "GPS not good enough, try again!", Toast.LENGTH_SHORT).show();
            return;
        };

        // Clear all waypoints and actions
        waypointList.clear();
        primarywaypointActions.clear();

        // Always create a new builder to prevent any leftover data from previous missions
        waypointMissionBuilder = new WaypointMission.Builder(); 

        // If the drone is not flying -> Arm -> Drone goes to 10m alt.
        
        // First waypoint, straight up from start to achieve two waypoints in total (required by DJI)
        double arm_lat = state.getAircraftLocation().getLatitude();
        double arm_lon = state.getAircraftLocation().getLongitude();
        float arm_alt = Math.max(state.getAircraftLocation().getAltitude(), 10.0f); // Go up to 10m or stay at current altitude if above 10m

        waypointList.add(new Waypoint(arm_lat, arm_lon, arm_alt)); // First waypoint makes it go up to h=10 or current height
        

        Waypoint mission_waypoint = new Waypoint(waypoint_lat, waypoint_lon, waypoint_alt);
        // If heading is specified. Set it to rotate on arrival
        if (waypoint_heading != null){
            WaypointAction rotate = new WaypointAction(WaypointActionType.ROTATE_AIRCRAFT, (int) waypoint_heading);
            mission_waypoint.addAction(rotate); // [-180, 180] Sets the drone yaw when arriving
        }

        waypointList.add(mission_waypoint);
        // ... (behåll din waypoint-logik ovanför) ...
        waypointMissionBuilder.waypointList(waypointList).waypointCount(waypointList.size());

        // Flytta stopp-logiken hit ner och använd dess callback
        WaypointMissionOperator operator = getWaypointMissionOperator();
        
        operator.stopMission(djiError -> {
            // struntar i att kontrollera if (djiError == null)
            
            if (djiError != null) {
                Log.d("DJI", "StopMission returnerade: " + djiError.getDescription() + " (Detta är normalt om inget kördes)");
            } else {
                Log.d("DJI", "Aktivt uppdrag stoppades framgångsrikt.");
            }

            // HÄR går vi vidare oavsett resultat!
            // Genom att ligga kvar inuti denna callback garanterar vi att vi väntat ut operatören.
            
            configWayPointMission();   
            addListener();         
            new android.os.Handler(android.os.Looper.getMainLooper()).postDelayed(() -> {
                Log.d("DJI", "Försöker ladda upp nytt uppdrag...");
                uploadWayPointMission(); 
                startWaypointMission();
            }, 500);

        });
    }



        /**
     * This function rotates the Gimbal
     * If successful it sends "task complete" to the backend, else it sends "task failed".
     * It awaits the specified duration before it send "task complete" or "task failed"
     */
    public void angleCamera(float pitch, float yaw, float transitionTime, String missionID, int taskIndex) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        // Create the rotation-object, didn't include .roll() since its prob. not needed
        Rotation rotation = new Rotation.Builder()
                // All below are methods of Builder(), chained for readability
                .pitch(pitch)
                .yaw(yaw)
                .mode(RotationMode.ABSOLUTE_ANGLE)
                .time(transitionTime) // To be able to pan an area
                .build();

        aircraft.getGimbal().rotate(rotation, djiError -> {
            if (djiError == null) {
                // Wait for the duration, then send task-complete if successful
                new android.os.Handler().postDelayed(() -> {
                    if (MessageHandler.getInstance() != null) {
                        MessageHandler.getInstance().taskComplete(currentMissionID, currentTaskIndex);
                    }
                    Log.d("DJI", "Successfully rotated camera | Task index: " + currentTaskIndex);
                }, (long) (transitionTime * 1000)); // convert s -> ms
            } else {
                if (MessageHandler.getInstance() != null) {
                    MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex);
                }
                Log.e("DJI", "Failed to rotate camera: " + djiError.getDescription());
            }
        });
    }

    public void stopCameraRotation(String missionID, int taskIndex) {
        if (aircraft == null || aircraft.getGimbal() == null || currentGimbalAttitude == null) {
            Log.e("DJI", "Gimbal or attitude not available");
            return;
        }

        // Stop it at current position
        Rotation stopRotation = new Rotation.Builder()
                .pitch(currentGimbalAttitude.getPitch())
                .yaw(currentGimbalAttitude.getYaw())
                .mode(RotationMode.ABSOLUTE_ANGLE)
                .time(0.1f) // Snabb inbromsning
                .build();

        aircraft.getGimbal().rotate(stopRotation, djiError -> {
            if (djiError == null) {
                Log.d("DJI", "Gimbal movement stopped at current attitude");
            }
        });
    }



    /**
     * This function starts the specified soundtrack playing on the speaker
     * If duration is specified it will automatically shut of after that time
     * If duration is NOT specified it will keep playing until "stop_task" is received
     */

    public void playAudio(String fileName, float volume, Integer durationSeconds, String missionID, int taskIndex) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        // Validate there is a speaker
        if (!AudioFileMapping.hasSpeaker(aircraft)) {
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex);
            }
            Log.e("DJI", "Speaker accessory not connected");
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex);
            }
            return;
        }

        if (audioIndexCache.isEmpty()) {
            AudioFileMapping.buildSpeakerCapabilitiesAndCache(aircraft, audioIndexCache);
        }

        Integer fileIndex = AudioFileMapping.getCachedFileIndex(audioIndexCache, fileName);

        if (fileIndex == null) {
            // messageHandler.sendTaskFailed(missionID, taskIndex, "Audio file not found in cache: " + fileName); ###########
            Log.e("DJI", "Audio file not found in cache: " + fileName);
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex);
            }
            return;
        }

        Speaker speaker = aircraft.getAccessoryAggregation().getSpeaker();


        // Set volume and play file
        speaker.setVolume((int) (volume * 100), djiError -> {
            // Continue ever if error with setting volume

            // The file will repeat till duration is reached or receiving "end_task"
            speaker.setPlayMode(PlayMode.REPEAT_SINGLE, error -> {
                if (error == null) {

                    speaker.play(fileIndex, (DJIError djiError1) -> {
                        if (djiError1 == null) {
                            Log.d("DJI", "Playing: " + fileName);

                            if (durationSeconds == null) {
                                if (MessageHandler.getInstance() != null) {
                                    MessageHandler.getInstance().taskComplete(currentMissionID, currentTaskIndex);
                                }
                                return;
                            }

                            // Play for specified duration
                            new android.os.Handler(android.os.Looper.getMainLooper()).postDelayed(() -> {

                                speaker.stop(stopError -> {
                                    Log.d("DJI", "Done playing audio | Task Index:" + currentTaskIndex);
                                    if (MessageHandler.getInstance() != null) {
                                        MessageHandler.getInstance().taskComplete(currentMissionID, currentTaskIndex);
                                    }
                                });

                            }, (long) (durationSeconds * 1000));
                        } else {
                            Log.e("DJI", "Play failed: " + djiError1.getDescription());
                            if (MessageHandler.getInstance() != null) {
                                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex);
                            }
                        }
                    });
                } else {
                    Log.e("DJI", "Failed to set play mode: " + error.getDescription());
                }
            });
        });
    }

    public void stopAudio(String missionID, int taskIndex) {
        boolean hasSpeaker = aircraft.getAccessoryAggregation() != null && aircraft.getAccessoryAggregation().getSpeaker() != null;
        if (!hasSpeaker) {
            Log.e("DJI", "Speaker accessory not connected");
            return;
        }

        Speaker speaker = aircraft.getAccessoryAggregation().getSpeaker();
        speaker.stop(stopError -> {
            if (stopError == null) {
                Log.d("DJI", "Audio stopped successfully");
            } else {
                Log.e("DJI", "Failed to stop audio: " + stopError.getDescription());
            }
        });
    }

    /**
     * This function turns on the spotlight with the specified brightness.
     * If durationSeconds <= 0, the spotlight stays on indefinitely.
     */
    public void activateSpotlight(float brightness, Integer durationSeconds, String missionID, int taskIndex) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        if (aircraft.getAccessoryAggregation() != null && aircraft.getAccessoryAggregation().getSpotlight() != null) {
            Spotlight spotlight = aircraft.getAccessoryAggregation().getSpotlight();
            
            // First, enable the spotlight
            spotlight.setEnabled(true, enableError -> {
                if (enableError == null) {
                    // Then set the brightness
                    spotlight.setBrightness((int) (brightness * 100), djiError -> {
                        if (djiError == null) {
                            Log.d("DJI", "Spotlight activated with brightness: " + brightness);
                            if (durationSeconds == null) {
                                // No duration specified, will stay on till "stop_task" is received
                                return;
                            }
                            // Finite duration - schedule deactivation
                            new android.os.Handler(android.os.Looper.getMainLooper()).postDelayed(() -> {
                                // Disable the spotlight
                                spotlight.setEnabled(false, disableError -> {
                                    if (disableError == null) {
                                        Log.d("DJI", "Spotlight deactivated after duration");
                                        if (MessageHandler.getInstance() != null) {
                                            MessageHandler.getInstance().taskComplete(currentMissionID, currentTaskIndex);
                                        }
                                    } else {
                                        Log.e("DJI", "Failed to disable spotlight: " + disableError.getDescription());
                                    }
                                });
                            }, (long) (durationSeconds * 1000));
                        } else {
                            Log.e("DJI", "Failed to set spotlight brightness: " + djiError.getDescription());
                        }
                    });
                } else {
                    Log.e("DJI", "Failed to enable spotlight: " + enableError.getDescription());
                }
            });
        } else {
            Log.e("DJI", "Spotlight accessory not connected");
        }
    }

    public void deactivateSpotlight(String missionID, int taskIndex) {
        if (aircraft.getAccessoryAggregation() != null && aircraft.getAccessoryAggregation().getSpotlight() != null) {
            Spotlight spotlight = aircraft.getAccessoryAggregation().getSpotlight();
            spotlight.setEnabled(false, disableError -> {
                if (disableError == null) {
                    Log.d("DJI", "Spotlight deactivated successfully");
                } else {
                    Log.e("DJI", "Failed to deactivate spotlight: " + disableError.getDescription());
                }
            });
        } else {
            Log.e("DJI", "Spotlight accessory not connected");
        }
    }

    public void activateLED(String ledType, Integer durationSeconds, String missionID, int taskIndex) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        String normalizedType = ledType == null ? "" : ledType.trim().toLowerCase();

        CommonCallbacks.CompletionCallback activationCallback = djiError -> {
            if (djiError == null) {
                Log.d("DJI", "LED " + ledType + " activated");
                if (durationSeconds == null || durationSeconds <= 0) {
                    return;
                }
                new android.os.Handler(android.os.Looper.getMainLooper()).postDelayed(() ->
                                deactivateLED(ledType, missionID, taskIndex),
                        (long) (durationSeconds * 1000)
                );
            } else {
                Log.e("DJI", "Failed to activate LED " + ledType + ": " + djiError.getDescription());
            }
        };

        if ("beacon".equals(normalizedType)) {
            if (aircraft != null
                    && aircraft.getAccessoryAggregation() != null
                    && aircraft.getAccessoryAggregation().getBeacon() != null) {
                aircraft.getAccessoryAggregation().getBeacon().setEnabled(true, activationCallback);
            } else {
                Log.e("DJI", "Beacon accessory not connected");
            }
            return;
        }

        if (controller == null) {
            Log.e("DJI", "Flight controller not available");
            return;
        }

        controller.getLEDsEnabledSettings(new CommonCallbacks.CompletionCallbackWith<LEDsSettings>() {
            @Override
            public void onSuccess(LEDsSettings currentSettings) {
                boolean frontOn = currentSettings != null && currentSettings.areFrontLEDsOn();
                boolean rearOn = currentSettings != null && currentSettings.areRearLEDsOn();
                boolean beaconOn = currentSettings != null && currentSettings.areBeaconsOn();
                boolean statusOn = currentSettings != null && currentSettings.isStatusIndicatorOn();

                switch (normalizedType) {
                    case "front":
                        frontOn = true;
                        Log.d("DJI", "Front LED will be activated");
                        break;
                    case "rear":
                        rearOn = true;
                        Log.d("DJI", "Rear LED will be activated");
                        break;
                    default:
                        Log.e("DJI", "Unknown LED type: " + ledType);
                        return;
                }

                LEDsSettings settings = new LEDsSettings.Builder()
                        .frontLEDsOn(frontOn)
                        .rearLEDsOn(rearOn)
                        .beaconsOn(beaconOn)
                        .statusIndicatorOn(statusOn)
                        .build();
                controller.setLEDsEnabledSettings(settings, activationCallback);
            }

            @Override
            public void onFailure(DJIError djiError) {
                Log.e("DJI", "Failed to read current LED settings: " + djiError.getDescription());
            }
        });
    }

    public void deactivateLED(String ledType, String missionID, int taskIndex) {
        String normalizedType = ledType == null ? "" : ledType.trim().toLowerCase();

        CommonCallbacks.CompletionCallback disableCallback = djiError -> {
            if (djiError == null) {
                Log.d("DJI", "LED " + ledType + " deactivated");
            } else {
                Log.e("DJI", "Failed to deactivate LED " + ledType + ": " + djiError.getDescription());
            }
        };

        if ("beacon".equals(normalizedType)) {
            if (aircraft != null
                    && aircraft.getAccessoryAggregation() != null
                    && aircraft.getAccessoryAggregation().getBeacon() != null) {
                aircraft.getAccessoryAggregation().getBeacon().setEnabled(false, disableCallback);
            } else {
                Log.e("DJI", "Beacon accessory not connected");
            }
            return;
        }

        if (controller == null) {
            Log.e("DJI", "Flight controller not available");
            return;
        }

        controller.getLEDsEnabledSettings(new CommonCallbacks.CompletionCallbackWith<LEDsSettings>() {
            @Override
            public void onSuccess(LEDsSettings currentSettings) {
                boolean frontOn = currentSettings != null && currentSettings.areFrontLEDsOn();
                boolean rearOn = currentSettings != null && currentSettings.areRearLEDsOn();
                boolean beaconOn = currentSettings != null && currentSettings.areBeaconsOn();
                boolean statusOn = currentSettings != null && currentSettings.isStatusIndicatorOn();

                switch (normalizedType) {
                    case "front":
                        frontOn = false;
                        break;
                    case "rear":
                        rearOn = false;
                        break;
                    default:
                        Log.e("DJI", "Unknown LED type: " + ledType);
                        return;
                }

                LEDsSettings settings = new LEDsSettings.Builder()
                        .frontLEDsOn(frontOn)
                        .rearLEDsOn(rearOn)
                        .beaconsOn(beaconOn)
                        .statusIndicatorOn(statusOn)
                        .build();

                controller.setLEDsEnabledSettings(settings, disableCallback);
            }

            @Override
            public void onFailure(DJIError djiError) {
                Log.e("DJI", "Failed to read current LED settings: " + djiError.getDescription());
            }
        });
    }

    public void goHome(String missionID, int taskIndex) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        stopAllTasks();
        goingHome();

    }


    /**
     * Stops all current tasks, including waypoint missions, audio, gimbal movements etc.
     * This is used when the user presses the "Abort" button, to immediately stop all drone activity and hover in current position.
     */
    public void stopAllTasks() {
        // Stop waypoint mission
        abortWaypointMission();

        // Stop audio if playing
        stopAudio(currentMissionID, currentTaskIndex);

        // Stop gimbal movement by resetting to current position (this is a workaround since DJI SDK does not provide a direct method to stop gimbal movement)
        stopCameraRotation(currentMissionID, currentTaskIndex);

        // Deactivate spotlight if active
        deactivateSpotlight(currentMissionID, currentTaskIndex);

        // Deactivate beacons
        deactivateLED("beacon", currentMissionID, currentTaskIndex);

        if (MessageHandler.getInstance() != null) {
            MessageHandler.getInstance().allTasksAborted(currentMissionID);
        }
    }


    /**
     * This function is used to stop a specific task when "stop_task" is received from the backend.
     * It checks the task type and calls the appropriate stop function for that task.
     */
    public void stopTask(String missionID, int taskIndex, String taskType) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;
        switch (taskType) {
            case "goTo":
                abortWaypointMission();
                break;
            case "playAudio":
                stopAudio(missionID, taskIndex);
                break;
            case "angleCamera":
                stopCameraRotation(missionID, taskIndex);
                break;
            case "spotlight":
                deactivateSpotlight(missionID, taskIndex);
                break;
            case "led_rear":
                // For LED we would ideally want to specify which LED to turn off, but for simplicity we can just try turning off all
                deactivateLED("rear", missionID, taskIndex);
            case "led_front":
                deactivateLED("front", missionID, taskIndex);
                break;
            case "led_beacon":
                deactivateLED("beacon", missionID, taskIndex);
                break;
            // Add cases for other task types as needed
            default:
                Log.e("DJI", "Unknown task type: " + taskType);
        }
        if (MessageHandler.getInstance() != null) {
            MessageHandler.getInstance().taskAborted(missionID, taskIndex, taskType);
        }
    }


    public void land(String missionID, int taskIndex) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        stopAllTasks();

        controller.startLanding(djiError -> {
            if (djiError == null){
                Log.d("DJI", "Landing initiated...");
            } else{
                Log.e("DJI", "Failed to start landing: " + djiError.getDescription());
            }
        });
    }


    /**
     * ConfigWayPointMission() extends the onArm function and builds the characteristics of the waypoint mission.
     * We set finishedAction and headingMode to what was defined in onArm()
     * We continue to define alla actions for our waypoint (test origin) in primaryWaypointActions
     * The list is connected to our main waypoint on index 1
     * After all configurations are complete the mission is ready to be uploaded to the drone
     *
     * @mFinishedAction Value of what behavior we want the drone to have after completing the final waypoint action
     * @mHeadingMode Value of what behavior we want when the drone flies between waypoints
     * @mSpeed The operating speed of the drone between the waypoints. In this case, auto and max is
     * set to the same for simplicity
     * @mFlightPathMode Defines how to fly between waypoints, either in a curve or straight(NORMAL).
     * In this case, NORMAL is used
     */
    private void configWayPointMission(){
        float mSpeed = 6.0f;
        // Vi behöver inte kolla om den är null längre eftersom vi skapar den i GoToWaypoint
        waypointMissionBuilder.finishedAction(mFinishedAction)
                .headingMode(mHeadingMode)
                .autoFlightSpeed(mSpeed)
                .maxFlightSpeed(mSpeed)
                .flightPathMode(mFlightPathMode);

        DJIError error = getWaypointMissionOperator().loadMission(waypointMissionBuilder.build());
        if (error != null) {
            Toast.makeText(getContext(), "loadWaypoint failed in stage config " + error.getDescription(), Toast.LENGTH_SHORT).show();
        } else {
            Log.d("DJI", "Mission loaded successfully in config stage, ready to upload");
        }
    }

    /**
     * Sends mission details to the drone after everything has been configured
     */
    private void uploadWayPointMission(){
        Log.d("DJI", "Uploading mission to drone...");
        getWaypointMissionOperator().uploadMission(error -> {
            if (error == null) {
                Toast.makeText(getContext(), "Mission upload successfully!", Toast.LENGTH_SHORT).show();
                Log.d("DJI", "Mission uploaded successfully, starting mission...");
            } else {
                Toast.makeText(getContext(), "Mission upload failed, error: " + error.getDescription() + " retrying...", Toast.LENGTH_SHORT).show();
                Log.e("DJI", "Mission upload failed: " + error.getDescription() + ", retrying...");
                getWaypointMissionOperator().retryUploadMission(null);
            }
        });
    }

    /**
     * Starts the waypoint mission. If a mission is uploaded, the drone will ACTUALLY TAKE OFF when this
     * method is called.
     */
    public void startWaypointMission(){
        getWaypointMissionOperator().startMission(error -> Toast.makeText(getContext(), "Mission Start: " + (error == null ? "Successfully" : error.getDescription()), Toast.LENGTH_SHORT).show());
    }

    /**
     * Function for terminating current waypoint mission.
     * This function is directly used when the Abort button is pressed
     * When called, the drone exits the waypoint mission and hovers in current position with manual controls activated
     */
    public void abortWaypointMission(){
        getWaypointMissionOperator().stopMission(error -> Toast.makeText(getContext(), "Mission Stop: " + (error == null ? "Successfully" : error.getDescription()), Toast.LENGTH_SHORT).show());
    }

    /**
     * Function to use when everything is done and the test is completed.
     * Waypointmode is stopped and the drone returns to home to start position.
     */
    public void endWaypointMission(){
        abortWaypointMission();
        goingHome();
    }


    /**
     * Implements the DJI function startGoHome.
     * Is called when test is complete
     */
    public void goingHome(){
        controller.startGoHome(djiError -> {
            if (djiError == null){
                Toast.makeText(getContext(), "Returning... :)", Toast.LENGTH_SHORT).show();
            } else{
                Toast.makeText(getContext(), djiError.getDescription(), Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Deprecated
    void setHomeLocationUsingAircraftCurrentLocation(){
        controller.setHomeLocationUsingAircraftCurrentLocation(djiError -> {
            if (djiError == null){
                Toast.makeText(getCoordinatesActivity(), "Home set :)", Toast.LENGTH_SHORT).show();
            } else{
                Toast.makeText(getCoordinatesActivity(), djiError.getDescription(), Toast.LENGTH_SHORT).show();
            }
        });
    }

    /**
     * See the DJI documentation for the Google Maps Demo app.
     */
    public void addListener() {
    if (getWaypointMissionOperator() != null && eventNotificationListener != null) {
        // First remove, to avoid cuplicates
        getWaypointMissionOperator().removeListener(eventNotificationListener);
        getWaypointMissionOperator().addListener(eventNotificationListener);
        }
    }

    /**
     * See the DJI documentation for the Google Maps Demo app.
     */
    public void removeListener() {
        if (getWaypointMissionOperator() != null) {
            getWaypointMissionOperator().removeListener(eventNotificationListener);
        }
    }
}
