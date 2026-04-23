package com.dji.sdk.sample;

import static dji.midware.data.manager.P3.ServiceManager.getContext;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
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
    private static final int SPEAKER_FILELIST_MAX_RETRIES = 3;
    private static final long SPEAKER_FILELIST_RETRY_DELAY_MS = 500L;
    // Near-ground thresholds used to detect the landing pause before final touchdown.
    private static final float LANDING_CONFIRMATION_ALTITUDE_METERS = 0.6f;
    private static final float LANDING_HOVER_VERTICAL_SPEED_MPS = 0.2f;

    private final List<Waypoint> waypointList = new ArrayList<>(); //List for storing waypoint / -s
    private final List<WaypointAction> primarywaypointActions = new ArrayList<>();
    public static WaypointMission.Builder waypointMissionBuilder;
    private WaypointMissionOperator waypointMissionOperator;
    private WaypointMissionFinishedAction mFinishedAction = WaypointMissionFinishedAction.NO_ACTION;
    private WaypointMissionHeadingMode mHeadingMode = WaypointMissionHeadingMode.USING_WAYPOINT_HEADING;
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
    // Prevents sending repeated confirmLanding calls while waiting for callback/state change.
    private volatile boolean landingConfirmationRequested = false;
    // Tracks if a go_home/land task should send taskComplete once the aircraft is actually down.
    private boolean landingTaskCompletionPending = false;
    private boolean landingTaskWasAirborne = false;
    private String landingCompletionMissionID = null;
    private int landingCompletionTaskIndex = -1;

    private Map<String, Integer> audioIndexCache = new ArrayMap<>();

    private Attitude currentGimbalAttitude;


    private DJIFlightManager(){
        this.aircraft = (Aircraft) DJISDKManager.getInstance().getProduct();
        this.controller = aircraft.getFlightController();
        controller.setStateCallback(new FlightControllerState.Callback() {
            @Override
            public void onUpdate(@NonNull FlightControllerState flightControllerState) {
                state = flightControllerState;
                handleLandingConfirmationIfSafe(flightControllerState);
                maybeCompleteLandingTask(flightControllerState);
            }
        });
        aircraft.getBattery().setStateCallback(new BatteryState.Callback() {
            @Override
            public void onUpdate(BatteryState batteryState) {
                DJIFlightManager.this.batteryState = batteryState;
            }
        });

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
                // Intentionally left empty.
            }
            @Override
            public void onExecutionStart() {
                Log.d("DJI", "Mission execution started for task " + currentTaskIndex + " mission " + currentMissionID);
            }
            @Override
            public void onExecutionFinish(@Nullable final DJIError error) {
                if (error == null) {
                   
                    Log.d("DJI", "Waypoint reached for task: " + currentTaskIndex);
                    
                    if (MessageHandler.getInstance() != null) {
                        MessageHandler.getInstance().taskComplete(currentMissionID, currentTaskIndex);
                    }
                } else {
                    Log.e("DJI", "Mission failed: " + error.getDescription());
                    if (MessageHandler.getInstance() != null) {
                        MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, error.getDescription());
                    }
                }
            }
        };
    }

    // Handles DJI landing-confirmation flow: when near ground and safe, confirm final touchdown.
    private void handleLandingConfirmationIfSafe(@NonNull FlightControllerState flightControllerState) {
        if (!flightControllerState.isLandingConfirmationNeeded()) {
            landingConfirmationRequested = false;
            return;
        }

        if (landingConfirmationRequested) {
            return;
        }

        dji.common.flightcontroller.LocationCoordinate3D location = flightControllerState.getAircraftLocation();
        if (location == null) {
            return;
        }

        float altitude = location.getAltitude();
        float verticalSpeed = flightControllerState.getVelocityZ();
        boolean nearGroundHover = altitude <= LANDING_CONFIRMATION_ALTITUDE_METERS
                && Math.abs(verticalSpeed) <= LANDING_HOVER_VERTICAL_SPEED_MPS;

        if (!nearGroundHover) {
            return;
        }

        String landingProtectionState = getLandingProtectionStateName(flightControllerState);
        if (!isSafeToLand(landingProtectionState)) {
            Log.w("DJI", "Landing confirmation paused. Landing spot not safe yet. state=" + landingProtectionState + " alt=" + altitude + " vz=" + verticalSpeed);
            return;
        }

        landingConfirmationRequested = true;
        controller.confirmLanding(djiError -> {
            if (djiError == null) {
                Log.i("DJI", "Landing confirmed after near-ground safety check.");
            } else {
                Log.e("DJI", "confirmLanding failed: " + djiError.getDescription());
                landingConfirmationRequested = false;
            }
        });
    }

    private String getLandingProtectionStateName(@NonNull FlightControllerState flightControllerState) {
        try {
            Object stateObj = flightControllerState.getClass().getMethod("getLandingProtectionState").invoke(flightControllerState);
            if (stateObj != null) {
                return stateObj.toString().toUpperCase();
            }
        } catch (Exception e) {
            Log.w("DJI", "Landing protection state unavailable in this SDK/aircraft: " + e.getMessage());
        }
        return "UNKNOWN";
    }

    private boolean isSafeToLand(String landingProtectionState) {
        if (landingProtectionState == null || landingProtectionState.isEmpty()) {
            return false;
        }

        String stateName = landingProtectionState.toUpperCase();
        if (stateName.contains("NOT_SAFE") || stateName.contains("UNSAFE")) {
            return false;
        }

        if (stateName.contains("SAFE_TO_LAND") || stateName.equals("SAFE")) {
            return true;
        }

        // If the landing-protection API is unavailable, avoid blocking supported drones forever.
        return stateName.equals("UNKNOWN");
    }

    // Called when land/go_home starts so completion is reported after touchdown, not on command start.
    private synchronized void startLandingTaskCompletionTracking(String missionID, int taskIndex) {
        landingTaskCompletionPending = true;
        landingCompletionMissionID = missionID;
        landingCompletionTaskIndex = taskIndex;
        landingTaskWasAirborne = state != null && state.isFlying();
    }

    private synchronized void clearLandingTaskCompletionTracking() {
        landingTaskCompletionPending = false;
        landingTaskWasAirborne = false;
        landingCompletionMissionID = null;
        landingCompletionTaskIndex = -1;
    }

    // Emits taskComplete once after the aircraft transitions from airborne to landed.
    private synchronized void maybeCompleteLandingTask(@NonNull FlightControllerState flightControllerState) {
        if (!landingTaskCompletionPending) {
            return;
        }

        boolean isFlying = flightControllerState.isFlying();
        if (isFlying) {
            landingTaskWasAirborne = true;
            return;
        }

        // Avoid reporting complete while landing protection is still waiting for confirmation.
        if (flightControllerState.isLandingConfirmationNeeded()) {
            return;
        }

        if (!landingTaskWasAirborne) {
            return;
        }

        String missionID = landingCompletionMissionID;
        int taskIndex = landingCompletionTaskIndex;
        clearLandingTaskCompletionTracking();

        if (MessageHandler.getInstance() != null) {
            MessageHandler.getInstance().taskComplete(missionID, taskIndex);
        }
        Log.i("DJI", "Landing task completed after touchdown. mission=" + missionID + " task=" + taskIndex);
    }

    /**
     * Returns the instance of the FlightManager class. If the FlightManger is not
     * instantiated, it is.
     * @return The instance of FlightManager
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

    /**
     * Fetches the current telemetry data from the drone and returns it as a DroneAdapter.Telemetry object.
     * If any critical piece of telemetry data is unavailable (e.g. GPS location), this method returns null to indicate that telemetry cannot be provided at this time.
    */
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
            getRegistrationData(new CommonCallbacks.CompletionCallbackWith<DroneAdapter.RegistrationData>() {
                @Override
                public void onSuccess(DroneAdapter.RegistrationData registrationData) {
                }

                @Override
                public void onFailure(DJIError djiError) {
                    Log.e("DJI", "Failed to fetch registration data for telemetry: " + (djiError != null ? djiError.getDescription() : "unknown error"));
                }
            });
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

        telemetry.batteryPercent = batteryState == null ? 0 : batteryState.getChargeRemainingInPercent();
        if (telemetry.batteryPercent < 0) {
            telemetry.batteryPercent = 0;
        }

        return telemetry;
    }

    public CoordinatesActivity getCoordinatesActivity() {
        return coordinatesActivity;
    }

    private interface RegistrationStepDone {
        void done();
    }

    /**
     * Fetches the registration data for the connected drone, including its type, model, unique ID, and capabilities.
     * The data is returned asynchronously via the provided callback.
     * If fetching the registration data fails for any reason, the callback's onFailure method is called with a reason string.
     */
    public void getRegistrationData(CommonCallbacks.CompletionCallbackWith<DroneAdapter.RegistrationData> callback) {
        BaseProduct product = DJISDKManager.getInstance().getProduct();
        if (!(product instanceof Aircraft)) {
            Log.e("DJI", "getRegistrationData failed: No drone connected.");
            if (callback != null) {
                callback.onFailure(DJIError.COMMON_DISCONNECTED);
            }
            return;
        }

        this.aircraft = (Aircraft) product;
        DroneAdapter.RegistrationData registrationData = createRegistrationDataSnapshot(this.aircraft);
        // Chain async calls to fetch drone ID, camera capabilities, LED capabilities, and speaker capabilities in sequence
        // then return the complete registration data in the callback
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

    /**
     * Fetches the drone model and updates the registration data object.
     * Uses cached model if available to avoid unnecessary SDK calls.
     */
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
            registrationData.capabilities.camera = null;
            Log.d("DJI", "No camera detected on aircraft; camera capabilities set to null.");
            done.done();
            return;
        }

        dji.sdk.camera.Camera camera = aircraft.getCamera();
        if (camera == null) {
            registrationData.capabilities.camera = null;
            Log.d("DJI", "Camera instance unavailable; camera capabilities set to null.");
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
                    cameraCapabilities.diagonal_fov = (float) DJIDroneSpecs.getDiagonalFov(aircraft.getModel(), djiFov);
                    cameraCapabilities.aspect_ratio = (res.height > 0) ? (float) res.width / res.height : 0.0f;

                    registrationData.capabilities.camera = cameraCapabilities;
                    Log.d("DJI", "Camera capabilities updated: " + djiRes.name());
                } else {
                    applyFallbackCameraCapabilities(aircraft, registrationData);
                    Log.w("DJI", "Camera settings callback returned null; using fallback camera capabilities.");
                }
                done.done();
            }

            @Override
            public void onFailure(DJIError djiError) {
                Log.e("DJI", "Failed to get resolution/FOV: " + djiError.getDescription());
                applyFallbackCameraCapabilities(aircraft, registrationData);
                done.done();
            }
        });
    }

    private void applyFallbackCameraCapabilities(Aircraft aircraft, DroneAdapter.RegistrationData registrationData) {
        DroneAdapter.Capabilities.Camera fallback = new DroneAdapter.Capabilities.Camera();
        DJIDroneSpecs.Resolution defaultRes = DJIDroneSpecs.getDimensions(SettingsDefinitions.VideoResolution.UNKNOWN);
        fallback.resolution_width = defaultRes.width;
        fallback.resolution_height = defaultRes.height;
        fallback.diagonal_fov = (float) DJIDroneSpecs.getDiagonalFov(aircraft.getModel(), null);
        fallback.aspect_ratio = defaultRes.height > 0
                ? (float) defaultRes.width / defaultRes.height
                : 16.0f / 9.0f;
        registrationData.capabilities.camera = fallback;
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

        refreshSpeakerFileListWithRetry(aircraft, registrationData, speaker, done, 1);
    }

    private void refreshSpeakerFileListWithRetry(
            Aircraft aircraft,
            DroneAdapter.RegistrationData registrationData,
            Speaker speaker,
            RegistrationStepDone done,
            int attempt
    ) {
        speaker.refreshFileList(djiError -> {
            if (djiError == null) {
                fetchSpeakerCapabilities(aircraft, registrationData);
                done.done();
                return;
            }

            Log.w("DJI","Failed to refresh speaker file list (attempt " + attempt + "/" + SPEAKER_FILELIST_MAX_RETRIES + "): " + djiError.getDescription());

            if (attempt < SPEAKER_FILELIST_MAX_RETRIES) {
                Handler mainHandler = new Handler(Looper.getMainLooper());
                mainHandler.postDelayed(
                        () -> refreshSpeakerFileListWithRetry(
                                aircraft,
                                registrationData,
                                speaker,
                                done,
                                attempt + 1
                        ),
                        SPEAKER_FILELIST_RETRY_DELAY_MS
                );
                return;
            }

            Log.e("DJI", "All speaker file list refresh attempts failed. Returning current snapshot.");
            fetchSpeakerCapabilities(aircraft, registrationData);

            boolean emptyAudioList = registrationData.capabilities != null
                    && registrationData.capabilities.speaker != null
                    && (registrationData.capabilities.speaker.audio_files == null
                    || registrationData.capabilities.speaker.audio_files.length == 0);
            if (emptyAudioList) {
                Log.e("DJI", "All speaker refresh attempts failed and speaker audio_files is empty.");
            }

            done.done();
        });
    }


    /**
    * This method starts the process of setting up the waypoint mission and is called when
    * the user presses the Arm button.
    * First we check battery state to ensure the drone battery is above 20%, because lower values
    * can cause unstable mission start behavior.
    * Second, we check GPS strength before mission setup.
    * To ensure a clean mission, leftovers from previous missions are removed.
    * The method fetches the current aircraft coordinates and creates the first waypoint 10 m above.
    * Then it adds the mission target coordinates loaded either manually or from ATOS.
    * The waypoints are then put in a list by order of execution,
    * so our primary waypoint is added lastly.
    * Finally it calls configWaypointMission() to configure behavior and
    * uploadWaypointMission() to send the mission to the drone.
    */

    public void GoToWaypoint(double waypoint_lat, double waypoint_lon, float waypoint_alt, Integer waypoint_heading, String missionID, int taskIndex){
        // Persist these values because SDK callbacks execute asynchronously.
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        if (state == null) {
            String errorMessage = "Cannot start go_to: flight controller state is null";
            Log.e("DJI", errorMessage);
            showToastOnMainThread(errorMessage);
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, errorMessage);
            }
            return;
        }

        if (batteryState == null) {
            String errorMessage = "Cannot start go_to: battery state unavailable";
            Log.e("DJI", errorMessage);
            showToastOnMainThread(errorMessage);
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, errorMessage);
            }
            return;
        }

        // Check battery before takeoff to avoid unstable mission start.
        int batteryPercent = batteryState.getChargeRemainingInPercent();
        Log.d("DJI", "GoTo precheck batteryPercent=" + batteryPercent);
        if (batteryPercent <= 20) {
            String errorMessage = "Battery too low to start mission, needs to be above 20%. Is: " + batteryPercent + "%";
            Log.e("DJI", errorMessage);
            showToastOnMainThread(errorMessage, Toast.LENGTH_LONG);
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, errorMessage);
            }
            return;
        }

        // Checking GPS before start to prevent crashes during next stage (GPS can be 1-5 and NONE)
        GPSSignalLevel gpsSignalLevel = state.getGPSSignalLevel();
        Log.d("DJI", "GoTo precheck gpsSignalLevel=" + gpsSignalLevel + " value=" + gpsSignalLevel.value());
        if ((Integer) gpsSignalLevel.value() <= 1 || gpsSignalLevel == GPSSignalLevel.NONE){
            String errorMessage = "GPS not good enough, try again!";
            Log.e("DJI", errorMessage + " level=" + gpsSignalLevel + " value=" + gpsSignalLevel.value());
            showToastOnMainThread(errorMessage);
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, errorMessage);
            }
            return;
        };

        // Fetching the current location of the drone to be able to set the first waypoint 10m above it
        dji.common.flightcontroller.LocationCoordinate3D aircraftLocation = state.getAircraftLocation();
        if (aircraftLocation == null) {
            String errorMessage = "Cannot start go_to: aircraft location is unavailable";
            Log.e("DJI", errorMessage);
            showToastOnMainThread(errorMessage);
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, errorMessage);
            }
            return;
        }

        // Clear all waypoints and actions
        waypointList.clear();
        primarywaypointActions.clear();

        // Always create a new builder to prevent any leftover data from previous missions
        waypointMissionBuilder = new WaypointMission.Builder(); 

        // If the drone is not flying -> Arm -> Drone goes to 10m alt.
        
        // First waypoint, straight up from start to achieve two waypoints in total (required by DJI)
        double arm_lat = aircraftLocation.getLatitude();
        double arm_lon = aircraftLocation.getLongitude();
        float cruise_alt = Math.max(aircraftLocation.getAltitude(), Math.max(waypoint_alt + 5.0f, 15.0f));

        waypointList.add(new Waypoint(arm_lat, arm_lon, cruise_alt));

        waypointList.add(new Waypoint(waypoint_lat, waypoint_lon, cruise_alt));

        Waypoint mission_waypoint = new Waypoint(waypoint_lat, waypoint_lon, waypoint_alt);
        if (waypoint_heading != null) {
            int heading = waypoint_heading;
            if (heading > 180) heading = heading - 360;
            if (heading < -180) heading = heading + 360;
            mission_waypoint.addAction(new WaypointAction(WaypointActionType.ROTATE_AIRCRAFT, heading));
        } else if (state.getAttitude() != null) {
            double yaw = state.getAttitude().yaw;
            if (!Double.isNaN(yaw) && yaw >= -180 && yaw <= 180) {
                int heading = (int) Math.round(yaw);
                mission_waypoint.addAction(new WaypointAction(WaypointActionType.ROTATE_AIRCRAFT, heading));
            }
        }
        waypointList.add(mission_waypoint);
        waypointMissionBuilder.waypointList(waypointList).waypointCount(waypointList.size());

        WaypointMissionOperator operator = getWaypointMissionOperator();
        if (operator == null) {
            String errorMessage = "Cannot start go_to: WaypointMissionOperator unavailable";
            Log.e("DJI", errorMessage);
            showToastOnMainThread(errorMessage);
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, errorMessage);
            }
            return;
        }
        
        /**
         * Before configuring and uploading the new mission, we stop any active mission to ensure a clean state.
         */
        operator.stopMission(djiError -> {
            
            if (djiError != null) {
                Log.d("DJI", "stopMission returned: " + djiError.getDescription() + " (normal if no mission was running)");
            } else {
                Log.d("DJI", "Active mission stopped successfully.");
            }
            
            configWayPointMission();   
            addListener();         
            new android.os.Handler(android.os.Looper.getMainLooper()).postDelayed(() -> {
                Log.d("DJI", "Trying to upload new mission...");
                uploadWayPointMission();
            }, 2000);

        });
    }



        /**
     * This method rotates the Gimbal
     * If successful it sends "task complete" to the backend, else it sends "task failed".
      * It waits for the specified duration before sending "task complete" or "task failed".
     */
    public void angleCamera(float pitch, float yaw, float transitionTime, String missionID, int taskIndex) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        // Roll is intentionally unchanged; only pitch and yaw are used here.
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
                    MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, djiError.getDescription());
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
            .time(0.1f) // Quick brake to stop movement smoothly
                .build();

        aircraft.getGimbal().rotate(stopRotation, djiError -> {
            if (djiError == null) {
                Log.d("DJI", "Gimbal movement stopped at current attitude");
            }
        });
    }



    /**
     * This method starts the specified soundtrack playing on the speaker
        * If duration is specified it will automatically shut off after that time
     * If duration is NOT specified it will keep playing until "stop_task" is received
     */

    public void playAudio(String fileName, float volume, Integer durationSeconds, String missionID, int taskIndex) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        // Validate there is a speaker
        if (!AudioFileMapping.hasSpeaker(aircraft)) {
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, "Speaker accessory not connected");
            }
            Log.e("DJI", "Speaker accessory not connected");
            return;
        }

        if (audioIndexCache.isEmpty()) {
            AudioFileMapping.buildSpeakerCapabilitiesAndCache(aircraft, audioIndexCache);
        }

        Integer fileIndex = AudioFileMapping.getCachedFileIndex(audioIndexCache, fileName);

        if (fileIndex == null) {
            Log.e("DJI", "Audio file not found in cache: " + fileName);
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, "Audio file not found in cache: " + fileName);
            }
            return;
        }

        Speaker speaker = aircraft.getAccessoryAggregation().getSpeaker();


        // Set volume and play file
        speaker.setVolume((int) (volume * 100), djiError -> {
            // Continue even if setting volume fails.

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
                                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, djiError1.getDescription());
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
     * This method turns on the spotlight with the specified brightness.
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
                                if (MessageHandler.getInstance() != null) {
                                    MessageHandler.getInstance().taskComplete(currentMissionID, currentTaskIndex);
                                }
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
                                        if (MessageHandler.getInstance() != null) {
                                            MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, "Failed to disable spotlight: " + disableError.getDescription());
                                        }
                                    }
                                });
                            }, (long) (durationSeconds * 1000));
                        } else {
                            Log.e("DJI", "Failed to set spotlight brightness: " + djiError.getDescription());
                            if (MessageHandler.getInstance() != null) {
                                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, "Failed to set spotlight brightness: " + djiError.getDescription());
                            }
                        }
                    });
                } else {
                    Log.e("DJI", "Failed to enable spotlight: " + enableError.getDescription());
                    if (MessageHandler.getInstance() != null) {
                        MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, "Failed to enable spotlight: " + enableError.getDescription());
                    }
                }
            });
        } else {
            Log.e("DJI", "Spotlight accessory not connected");
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(missionID, taskIndex, "Spotlight accessory not connected");
            }
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

    /*
    * This method activates the specified LED type (front, rear, or beacon) for the given duration.
     * If durationSeconds <= 0, the LED stays on indefinitely until "stop_task" is received.
    */
    public void activateLED(String ledType, Integer durationSeconds, String missionID, int taskIndex) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;

        String normalizedType = ledType == null ? "" : ledType.trim().toLowerCase();

        CommonCallbacks.CompletionCallback activationCallback = djiError -> {
            if (djiError == null) {
                Log.d("DJI", "LED " + ledType + " activated");
                if (durationSeconds == null || durationSeconds <= 0) {
                    // No duration means the LED stays on indefinitely.
                    // Still report task complete so the backend can send the next task.
                    if (MessageHandler.getInstance() != null) {
                        MessageHandler.getInstance().taskComplete(missionID, taskIndex);
                    }
                    return;
                }
                new android.os.Handler(android.os.Looper.getMainLooper()).postDelayed(() -> {
                    // Deactivate and then report task complete to the backend
                    CommonCallbacks.CompletionCallback timedOffCallback = deactivateError -> {
                        if (deactivateError == null) {
                            Log.d("DJI", "LED " + ledType + " deactivated after duration");
                            if (MessageHandler.getInstance() != null) {
                                MessageHandler.getInstance().taskComplete(missionID, taskIndex);
                            }
                        } else {
                            Log.e("DJI", "Failed to deactivate LED " + ledType + " after duration: " + deactivateError.getDescription());
                            if (MessageHandler.getInstance() != null) {
                                MessageHandler.getInstance().taskFailed(missionID, taskIndex, deactivateError.getDescription());
                            }
                        }
                    };
                    if ("beacon".equals(normalizedType)) {
                        if (aircraft != null
                                && aircraft.getAccessoryAggregation() != null
                                && aircraft.getAccessoryAggregation().getBeacon() != null) {
                            aircraft.getAccessoryAggregation().getBeacon().setEnabled(false, timedOffCallback);
                        } else {
                            Log.e("DJI", "Beacon accessory not connected for timed deactivation");
                        }
                        return;
                    }
                    if (controller == null) {
                        Log.e("DJI", "Flight controller not available for timed LED deactivation");
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
                                case "front": frontOn = false; break;
                                case "rear":  rearOn  = false; break;
                                default:
                                    Log.e("DJI", "Unknown LED type for timed deactivation: " + ledType);
                                    return;
                            }
                            LEDsSettings settings = new LEDsSettings.Builder()
                                    .frontLEDsOn(frontOn).rearLEDsOn(rearOn)
                                    .beaconsOn(beaconOn).statusIndicatorOn(statusOn).build();
                            controller.setLEDsEnabledSettings(settings, timedOffCallback);
                        }
                        @Override
                        public void onFailure(DJIError djiError) {
                            Log.e("DJI", "Failed to read LED settings for timed deactivation: " + djiError.getDescription());
                            if (MessageHandler.getInstance() != null) {
                                MessageHandler.getInstance().taskFailed(missionID, taskIndex, djiError.getDescription());
                            }
                        }
                    });
                }, (long) (durationSeconds * 1000));
            } else {
                Log.e("DJI", "Failed to activate LED " + ledType + ": " + djiError.getDescription());
                if (MessageHandler.getInstance() != null) {
                    MessageHandler.getInstance().taskFailed(missionID, taskIndex, djiError.getDescription());
                }
            }
        };

        if ("beacon".equals(normalizedType)) {
            if (aircraft != null
                    && aircraft.getAccessoryAggregation() != null
                    && aircraft.getAccessoryAggregation().getBeacon() != null) {
                aircraft.getAccessoryAggregation().getBeacon().setEnabled(true, activationCallback);
            } else {
                Log.e("DJI", "Beacon accessory not connected");
                if (MessageHandler.getInstance() != null) {
                        MessageHandler.getInstance().taskFailed(missionID, taskIndex, "Beacon accessory not connected");
                }
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

    // Stop all tasks and go home
    public void goHome(String missionID, int taskIndex) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;
        startLandingTaskCompletionTracking(missionID, taskIndex);

        stopAllTasks();
        goingHome();

    }


    /**
     * Stops all current tasks, including waypoint missions, audio, gimbal movements etc.
     * This is used when the user presses the "Abort" button, to immediately stop all drone activity and hover in current position.
     */
    public void stopAllTasks() {
        // Stop waypoint mission
        try {
        abortWaypointMission();
        } catch (Exception e) {
            Log.e("DJI", "Error stopping waypoint mission: " + e.getMessage());
         }

        // Stop audio if playing
        try {
        stopAudio(currentMissionID, currentTaskIndex);
        } catch (Exception e) {
            Log.e("DJI", "Error stopping audio: " + e.getMessage());
         }

        // Stop gimbal movement by resetting to current position (this is a workaround since DJI SDK does not provide a direct method to stop gimbal movement)
        try {
            stopCameraRotation(currentMissionID, currentTaskIndex);
        } catch (Exception e) {
            Log.e("DJI", "Error stopping gimbal movement: " + e.getMessage());
        }

        // Deactivate spotlight if active
        try {
            deactivateSpotlight(currentMissionID, currentTaskIndex);
        } catch (Exception e) {
            Log.e("DJI", "Error deactivating spotlight: " + e.getMessage());
        }

        // Deactivate beacons
        try {
            deactivateLED("beacon", currentMissionID, currentTaskIndex);
        } catch (Exception e) {
            Log.e("DJI", "Error deactivating beacons: " + e.getMessage());
        }

        if (MessageHandler.getInstance() != null) {
            MessageHandler.getInstance().allTasksAborted(currentMissionID);
        }
    }


    /**
     * This method is used to stop a specific task when "stop_task" is received from the backend.
     * It checks the task type and calls the appropriate stop method for that task.
     */
    public void stopTask(String missionID, int taskIndex, String taskType) {
        this.currentMissionID = missionID;
        this.currentTaskIndex = taskIndex;
        switch (taskType) {
            case "go_to":
                abortWaypointMission();
                break;
            case "play_audio":
                stopAudio(missionID, taskIndex);
                break;
            case "angle_camera":
                stopCameraRotation(missionID, taskIndex);
                break;
            case "spotlight":
                deactivateSpotlight(missionID, taskIndex);
                break;
            case "led_rear":
                deactivateLED("rear", missionID, taskIndex);
                break;
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
        startLandingTaskCompletionTracking(missionID, taskIndex);

        stopAllTasks();

        controller.startLanding(djiError -> {
            if (djiError == null){
                Log.d("DJI", "Landing initiated...");
            } else{
                Log.e("DJI", "Failed to start landing: " + djiError.getDescription());
                clearLandingTaskCompletionTracking();
                if (MessageHandler.getInstance() != null) {
                    MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, djiError.getDescription());
                }
            }
        });
    }


    /**
     * ConfigWayPointMission() extends the onArm method and builds the characteristics of the waypoint mission.
     * We set finishedAction and headingMode to what was defined in onArm()
        * We continue to define all actions for our waypoint (test origin) in primaryWaypointActions
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
        waypointMissionBuilder.finishedAction(mFinishedAction)
                .headingMode(mHeadingMode)
                .autoFlightSpeed(mSpeed)
                .maxFlightSpeed(mSpeed)
                .flightPathMode(mFlightPathMode);

        // Logga alla waypoints innan upload
        Log.d("DJI", "=== WAYPOINT MISSION CONFIG ===");
        Log.d("DJI", "Number of waypoints: " + waypointList.size());
        for (int i = 0; i < waypointList.size(); i++) {
            Waypoint wp = waypointList.get(i);
            Log.d("DJI", "Waypoint " + i

                + " alt=" + wp.altitude
                + " actions=" + wp.waypointActions.size());
        }

        DJIError error = getWaypointMissionOperator().loadMission(waypointMissionBuilder.build());
        if (error != null) {
            Log.e("DJI", "loadMission failed: " + error.getDescription());
            showToastOnMainThread("loadWaypoint failed: " + error.getDescription());
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, "loadMission failed: " + error.getDescription());
            }
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
                showToastOnMainThread("Mission upload successfully!");
                Log.d("DJI", "Mission uploaded successfully, starting mission...");
                startWaypointMission();
            } else {
                Log.e("DJI", "Mission upload failed: " + error.getDescription());
                showToastOnMainThread("Mission upload failed: " + error.getDescription() + " retrying...");
                getWaypointMissionOperator().retryUploadMission(djiError -> {
                    if (djiError == null) {
                        Log.d("DJI", "Retry upload successful!");
                        startWaypointMission();
                    } else {
                        Log.e("DJI", "Retry upload failed: " + djiError.getDescription());
                        if (MessageHandler.getInstance() != null) {
                            MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, "Upload failed: " + djiError.getDescription());
                        }
                    }
                });
            }
        });
    }

    /**
     * Starts the waypoint mission. If a mission is uploaded, the drone will ACTUALLY TAKE OFF when this
     * method is called.
     */
    public void startWaypointMission(){
        WaypointMissionOperator operator = getWaypointMissionOperator();
        if (operator == null) {
            String message = "Mission Start: Operator unavailable";
            Log.e("DJI", message + " for mission " + currentMissionID + " task " + currentTaskIndex);
            showToastOnMainThread(message);
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, message);
            }
            return;
        }

        operator.startMission(error -> {
            if (error == null) {
                String message = "Mission Start: Successfully";
                Log.d("DJI", message + " for mission " + currentMissionID + " task " + currentTaskIndex);
                showToastOnMainThread(message);
                return;
            }

            String errorDescription = error.getDescription();
            String message = "Mission Start: " + errorDescription;
            Log.e("DJI", message + " for mission " + currentMissionID + " task " + currentTaskIndex);
            showToastOnMainThread(message);
            if (MessageHandler.getInstance() != null) {
                MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, errorDescription);
            }
        });
    }

    /**
    * Method for terminating current waypoint mission.
    * This method is directly used when the Abort button is pressed
    * When called, the drone exits the waypoint mission and hovers in current position with manual controls activated
    */
    public void abortWaypointMission(){
        WaypointMissionOperator operator = getWaypointMissionOperator();
        if (operator == null) {
            Log.e("DJI", "WaypointMissionOperator unavailable when trying to stop mission");
            showToastOnMainThread("Mission Stop: Operator unavailable");
            return;
        }

        operator.stopMission(
                error -> showToastOnMainThread(
                        "Mission Stop: " + (error == null ? "Successfully" : error.getDescription())
                )
        );
    }

    private void showToastOnMainThread(String message) {
        showToastOnMainThread(message, Toast.LENGTH_SHORT);
    }

    // Utility method to show a toast message on the main thread, since DJI callbacks may be on a background thread.
    private void showToastOnMainThread(String message, int duration) {
        Handler mainHandler = new Handler(Looper.getMainLooper());
        mainHandler.post(() -> {
            Context context = getContext();
            if (context == null) {
                Log.e("DJI", "Unable to show toast because context is null. message=" + message);
                return;
            }
            
            Toast.makeText(context.getApplicationContext(), message, duration).show();
        });
    }

    /**
    * Method to use when everything is done and the test is completed.
    * Waypoint mode is stopped and the drone returns to its home position.
    */
    public void endWaypointMission(){
        abortWaypointMission();
        goingHome();
    }


    /**
     * Implements the DJI method startGoHome.
     * Is called when test is complete
     */
    public void goingHome(){
        controller.startGoHome(djiError -> {
            if (djiError == null){
                showToastOnMainThread("Going Home...");
            } else{
                showToastOnMainThread(djiError.getDescription());
                clearLandingTaskCompletionTracking();
                if (MessageHandler.getInstance() != null) {
                    MessageHandler.getInstance().taskFailed(currentMissionID, currentTaskIndex, djiError.getDescription());
                }
            }
        });
    }

    @Deprecated
    void setHomeLocationUsingAircraftCurrentLocation(){
        controller.setHomeLocationUsingAircraftCurrentLocation(djiError -> {
            if (djiError == null){
                showToastOnMainThread("Home set");
            } else{
                showToastOnMainThread(djiError.getDescription());
            }
        });
    }

    /**
     * See the DJI documentation for the Google Maps Demo app.
     */
    public void addListener() {
    if (getWaypointMissionOperator() != null && eventNotificationListener != null) {
        // First remove, to avoid duplicates
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
