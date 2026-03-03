package com.dji.sdk.sample;

import android.util.Log;

public class DJIAdapter implements DroneAdapter {
    private static final String TAG = DJIAdapter.class.getSimpleName();
    private static final DJIAdapter INSTANCE = new DJIAdapter();

    private DJIAdapter() {
    }

    public static DJIAdapter getInstance() {
        return INSTANCE;
    }

    private DJIFlightManager flightManager() {
        return DJIFlightManager.getFlightManager();
    }

    @Override
    public void goTo(double lat, double lon, float alt, Integer heading, String missionID, int taskIndex) {
        flightManager().GoToWaypoint(lat, lon, alt, heading, missionID, taskIndex);
    }

    @Override
    public void angleCamera(float pitch, float yaw, Float transitionTime, String missionID, int taskIndex) {
        float safeTransitionTime = transitionTime == null ? 0f : transitionTime;
        flightManager().angleCamera(pitch, yaw, safeTransitionTime, missionID, taskIndex);
    }

    @Override
    public void goHome(String missionID, int taskIndex) {
        flightManager().goHome(missionID, taskIndex);
    }

    @Override
    public void playAudio(String file, float volume, Integer durationSeconds, String missionID, int taskIndex) {
        flightManager().playAudio(file, volume, durationSeconds, missionID, taskIndex);
    }

    @Override
    public void stopAudio(String missionID, int taskIndex) {
        flightManager().stopAudio(missionID, taskIndex);
    }

    @Override
    public void led(String type, Integer durationSeconds, String missionID, int taskIndex) {
        flightManager().activateLED(type, durationSeconds, missionID, taskIndex);
    }

    @Override
    public void deactivateLed(String type, String missionID, int taskIndex) {
        flightManager().deactivateLED(type, missionID, taskIndex);
    }

    @Override
    public void spotlight(float brightness, Integer durationSeconds, String missionID, int taskIndex) {
        int safeDurationSeconds = durationSeconds == null ? 0 : durationSeconds;
        flightManager().activateSpotlight(brightness, safeDurationSeconds, missionID, taskIndex);
    }

    @Override
    public void deactivateSpotlight(String missionID, int taskIndex) {
        flightManager().deactivateSpotlight(missionID, taskIndex);
    }

    @Override
    public void abortTask(String missionID, int taskIndex, String taskType) {
        flightManager().stopTask(missionID, taskIndex, taskType);
    }

    @Override
    public void stopAllTasks(String missionID, int taskIndex) {
        flightManager().stopAllTasks();
    }

    @Override
    public void land(String missionID, int taskIndex) {
        flightManager().land(missionID, taskIndex);
    }

    @Override
    public Telemetry getTelemetry() {
        return flightManager().getTelemetry();
    }

    @Override
    public RegistrationData getRegistrationData() {
        return flightManager().getRegistrationData();
    }

    @Override
    public void pushTaskComplete(String missionID, int taskIndex) {
        Log.d(TAG, "Task complete | missionID=" + missionID + " taskIndex=" + taskIndex);
    }

    @Override
    public void pushTaskFailed(String missionID, int taskIndex) {
        Log.e(TAG, "Task failed | missionID=" + missionID + " taskIndex=" + taskIndex);
    }
}
