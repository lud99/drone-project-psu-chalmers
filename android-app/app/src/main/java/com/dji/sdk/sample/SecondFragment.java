package com.dji.sdk.sample;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import com.dji.sdk.sample.databinding.FragmentSecondBinding;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import dji.common.error.DJIError;
import dji.common.flightcontroller.FlightControllerState;
import dji.common.mission.MissionState;
import dji.common.mission.waypoint.WaypointMissionDownloadEvent;
import dji.common.mission.waypoint.WaypointMissionExecutionEvent;
import dji.common.mission.waypoint.WaypointMissionUploadEvent;
import dji.sdk.camera.VideoFeeder;
import dji.sdk.codec.DJICodecManager;
import dji.sdk.mission.MissionControl;
import dji.sdk.mission.waypoint.WaypointMissionOperator;
import dji.sdk.mission.waypoint.WaypointMissionOperatorListener;
import dji.sdk.products.Aircraft;
import dji.sdk.sdkmanager.DJISDKManager;
import dji.sdk.sdkmanager.LiveStreamManager;
import dji.sdk.sdkmanager.LiveVideoBitRateMode;

public class SecondFragment extends Fragment{

    private FragmentSecondBinding binding;
    protected TextureView cameraTextureView;
    private CameraSurfaceTextureListener cameraSurfaceTextureListener;
    static final String TAG = MainActivity.class.getName();
    LiveStreamManager liveStreamManager;

    DJIFlightManager flightmanager;

    FlightControllerState state;

    private CameraController cameraController;

    private volatile boolean isWaypointMonitored = false;
    private final android.os.Handler statusHandler = new android.os.Handler(android.os.Looper.getMainLooper());
    private final Runnable statusUpdateRunnable = new Runnable() {
        @Override
        public void run() {
            if (!isAdded() || binding == null) {
                return;
            }
            statusUpdate();
            statusHandler.postDelayed(this, 750);
        }
    };

    @Override
    public View onCreateView(
            LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState
    ) {
        binding = FragmentSecondBinding.inflate(inflater, container, false);
        cameraController = CameraController.getInstance();

        flightmanager = DJIFlightManager.getFlightManager();

        cameraTextureView = binding.cameraTextureView;

        cameraSurfaceTextureListener = new CameraSurfaceTextureListener(getContext());


        Aircraft aircraft = (Aircraft) DJISDKManager.getInstance().getProduct();
        aircraft.getFlightController().setStateCallback(new FlightControllerState.Callback() {
            @Override
            public void onUpdate(@NonNull FlightControllerState flightControllerState) {
                state = flightControllerState;
            }
        });

        if (null != cameraTextureView){
            cameraTextureView.setSurfaceTextureListener(cameraSurfaceTextureListener);
        }

        initLiveStreamManager();

        flightmanager.addListener(); ///// Relevant for waypoint

        return binding.getRoot();

    }

    private void initLiveStreamManager() {
        liveStreamManager = DJISDKManager.getInstance().getLiveStreamManager();
        if (liveStreamManager != null){
            Toast.makeText(this.getContext(), "liveStreamManager available", Toast.LENGTH_SHORT).show();
            liveStreamManager.setLiveVideoBitRateMode(LiveVideoBitRateMode.AUTO);
            liveStreamManager.setLiveUrl(getString(R.string.stream_url));
            liveStreamManager.setAudioMuted(true);
        } else{
            Toast.makeText(this.getContext(), "liveStreamManager not available", Toast.LENGTH_LONG).show();
        }

    }

    /**
     * Binds the camera output from the drone to the TextureView that displays it.
     */
    private void onProductChange(){
        cameraController.initPreviewer(cameraTextureView, getContext(), cameraSurfaceTextureListener);
    }

    @Override
    public void onResume() {
        Log.e(TAG, "onResume");
        super.onResume();
        onProductChange();
        statusHandler.removeCallbacks(statusUpdateRunnable);
        statusHandler.post(statusUpdateRunnable);

        statusUpdate();

        if(cameraTextureView == null) {
            Log.e(TAG, "mVideoSurface is null");
        }

    }

    @Override
    public void onPause() {
        statusHandler.removeCallbacks(statusUpdateRunnable);
        super.onPause();
    }


    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);


        statusUpdate();

        binding.fitViewButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Set the correct size of the video playback
                statusUpdate();
                updateVideoSize(binding.cameraTextureView);
            }
        });

        binding.armButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(getStatus().equals(MissionState.READY_TO_UPLOAD) && !flightmanager.getState().isFlying()){
                    //startMonitoring();
                    //flightmanager.onArm();
                    statusUpdate();
                } else if(flightmanager.getState().isFlying()){
                    Toast.makeText(getContext(), "Can't upload in flight!", Toast.LENGTH_SHORT).show();
                } else{
                    Toast.makeText(getContext(), "Not ready to upload!", Toast.LENGTH_SHORT).show();
                }
            }
        });
        binding.startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Only want to allow start if waypoint mission is ready to execute and the drone is
                // stationary on the ground. Don't want to allow start while already airborne
                if (getStatus().equals(MissionState.READY_TO_EXECUTE) && !flightmanager.getState().isFlying()){
                    flightmanager.startWaypointMission();
                    statusUpdate();
                } else {
                    Toast.makeText(getContext(), "Not in ready state", Toast.LENGTH_SHORT).show();
                }

            }
        });
        binding.finishButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // We should only be able to finish a run if we are currently executing a run,
                // either if we are currently running or have temporarily paused execution
                if(getStatus().equals(MissionState.EXECUTING) || getStatus().equals(MissionState.EXECUTION_PAUSED)){
                    flightmanager.endWaypointMission();
                    statusUpdate();
                } else{
                    Toast.makeText(getContext(), "Not executing", Toast.LENGTH_SHORT).show();
                }
            }
        });

        binding.abortButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // We should only be able to abort a run if we are currently executing a run,
                // either if we are currently running or have temporarily paused execution
                if(getStatus().equals(MissionState.EXECUTING) || getStatus().equals(MissionState.EXECUTION_PAUSED)){
                    flightmanager.abortWaypointMission();
                    statusUpdate();
                } else{
                    Toast.makeText(getContext(), "Not executing", Toast.LENGTH_SHORT).show();
                }
            }
        });

        binding.pointDownButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                double lat = state.getAircraftLocation().getLatitude();
                double lng = state.getAircraftLocation().getLongitude();
                double alt = state.getAircraftLocation().getLatitude();

                float heading = flightmanager.getController().getCompass().getHeading();

                Toast.makeText(getContext(), "lat: "+lat, Toast.LENGTH_SHORT).show();
                Toast.makeText(getContext(), "lng: "+lng, Toast.LENGTH_SHORT).show();
                Toast.makeText(getContext(), "alt: "+alt, Toast.LENGTH_SHORT).show();
                Toast.makeText(getContext(), "heading: "+heading, Toast.LENGTH_SHORT).show();


                //Gimbal down
                boolean successful = CameraController.getInstance().gimbalDown(getContext());
                if (successful){
                    binding.pointDownButton.setText(R.string.pointed_down);
                }
            }
        });

        binding.gimbalRightButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                //Gimbal right

                SeekBar progress = binding.rotationBar;
                int rotation = progress.getProgress();
                //progress.setProgress(rotation+30);

                boolean successful = CameraController.getInstance().gimbalRight(getContext(),10);
                if (successful){
                    //change progressbar
                    progress.setProgress(rotation+10);
                }
            }
        });
        binding.gimbalLeftButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                //Gimbal left
                SeekBar progress = binding.rotationBar;
                int rotation = progress.getProgress();



                boolean successful = CameraController.getInstance().gimbalLeft(getContext(),10);
                if (successful){
                    //change progressbar
                    progress.setProgress(rotation-10);
                }

            }
        });

    }

    /**
     * Updates the status text in the top-left to the status of the drone's MissionWaypointOperator.
     */
    private void statusUpdate() {
        Boolean GoHomeStatus = flightmanager.getState().isGoingHome();
        Boolean FlightStatus = flightmanager.getState().isFlying();

        MissionControl missionControl = DJISDKManager.getInstance().getMissionControl();
        WaypointMissionOperator missionOperator = missionControl.getWaypointMissionOperator();

        if (missionOperator.getCurrentState().toString().equals("EXECUTING")){
            binding.statusTextView.setText(missionOperator.getCurrentState().toString());
            binding.statusTextView.setTextColor(ContextCompat.getColor(this.getContext(), R.color.green));
        } else if (missionOperator.getCurrentState().toString().equals("READY_TO_EXECUTE")) {
            binding.statusTextView.setText(missionOperator.getCurrentState().toString());
            binding.statusTextView.setTextColor(ContextCompat.getColor(this.getContext(), R.color.yellow));
        } else if (GoHomeStatus){
            binding.statusTextView.setText("GOING_HOME");
            binding.statusTextView.setTextColor(ContextCompat.getColor(this.getContext(), R.color.green));
        } else if (FlightStatus){
            binding.statusTextView.setText("IN_MANUAL_FLIGHT");
            binding.statusTextView.setTextColor(ContextCompat.getColor(this.getContext(), R.color.white));
        } else if (missionOperator.getCurrentState().toString().equals("READY_TO_UPLOAD")){
            binding.statusTextView.setText(missionOperator.getCurrentState().toString());
            binding.statusTextView.setTextColor(ContextCompat.getColor(this.getContext(), R.color.blue));
        } else {
            binding.statusTextView.setText(missionOperator.getCurrentState().toString());
            binding.statusTextView.setTextColor(ContextCompat.getColor(this.getContext(), R.color.white));
        }

    }
    /**
     * Use to check where in a mission the drone is and which buttons to allow
     * @return The current state of the mission
     */
    private MissionState getStatus() {
        MissionControl missionControl = DJISDKManager.getInstance().getMissionControl();
        WaypointMissionOperator missionOperator = missionControl.getWaypointMissionOperator();
        return missionOperator.getCurrentState();
    }

    /**
     * Updates the status text in the top-left to the status of the drone's MissionWaypointOperator
     * and includes the ability to add a custom message at the end
     * @param message The message to be included.
     */
    private void statusUpdate(String message){
        MissionControl missionControl = DJISDKManager.getInstance().getMissionControl();
        WaypointMissionOperator missionOperator = missionControl.getWaypointMissionOperator();
        binding.statusTextView.setText(String.format("%s%s", missionOperator.getCurrentState().toString(), message));
        if (missionOperator.getCurrentState().toString().equals("READY_TO_EXECUTE")){
            binding.statusTextView.setTextColor(ContextCompat.getColor(this.getContext(), R.color.green));
        }
        else{
            binding.statusTextView.setTextColor(ContextCompat.getColor(this.getContext(), R.color.yellow));
        }

    }

    /**
     * This method does not work, do not call it.
     * TODO Fix this method
     */
    private void startMonitoring(){
        MissionControl missionControl = DJISDKManager.getInstance().getMissionControl();
        WaypointMissionOperator missionOperator = missionControl.getWaypointMissionOperator();
        missionOperator.addListener(new WaypointMissionOperatorListener() {
            @Override
            public void onDownloadUpdate(@NonNull WaypointMissionDownloadEvent waypointMissionDownloadEvent) {
                statusUpdate();
            }

            @Override
            public void onUploadUpdate(@NonNull WaypointMissionUploadEvent waypointMissionUploadEvent) {
                statusUpdate();
            }

            @Override
            public void onExecutionUpdate(@NonNull WaypointMissionExecutionEvent waypointMissionExecutionEvent) {
                statusUpdate();
            }

            @Override
            public void onExecutionStart() {
                statusUpdate();
            }

            @Override
            public void onExecutionFinish(@Nullable DJIError djiError) {
                if (djiError == null) {
                    statusUpdate();
                } else {
                    statusUpdate(djiError.getDescription());
                }
            }
        });
    }

    /**
     * Forces an aspect ratio of 16:9 for the output centers it in the frame.
     * @param cameraTextureView The TextureView that holds the video
     */
    private void updateVideoSize(TextureView cameraTextureView){
        int height = binding.constraintLayout.getHeight();
        int width = binding.constraintLayout.getWidth();
        int newHeight = height;
        int newWidth = width;
        android.graphics.Matrix viewMatrix = new android.graphics.Matrix();

        double ratio = (double) width / height;
        double sixteen_nine = 1.77777778; // 16:9

        if (ratio > sixteen_nine) { // Video too wide
            newWidth = (int) (width * (sixteen_nine / ratio));
        } else { // Video too tall
            newHeight = (int) (height * (ratio / sixteen_nine));
        }

        cameraTextureView.getTransform(viewMatrix);
        viewMatrix.setScale((float) newWidth / width, (float) newHeight / height); // Set the new scale

        // Calculate translation to center the TextureView
        float dx = (width - newWidth) / 2f;
        float dy = (height - newHeight) / 2f;
        viewMatrix.postTranslate(dx, dy);

        cameraTextureView.setTransform(viewMatrix); // Apply the changes
    }
    @Override
    public void onDestroyView() {
        statusHandler.removeCallbacks(statusUpdateRunnable);
        super.onDestroyView();
        binding = null;

        flightmanager.removeListener();
    }

}