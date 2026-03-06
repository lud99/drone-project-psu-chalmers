package com.dji.sdk.sample;

import android.content.Context;
import android.util.Log;
import android.view.TextureView;
import android.widget.Toast;

import androidx.annotation.NonNull;

import dji.common.gimbal.GimbalState;
import dji.common.gimbal.Rotation;
import dji.common.gimbal.RotationMode;
import dji.common.product.Model;
import dji.sdk.base.BaseProduct;
import dji.sdk.camera.VideoFeeder;
import dji.sdk.codec.DJICodecManager;
import dji.sdk.gimbal.Gimbal;
import dji.sdk.sdkmanager.DJISDKManager;
import java.lang.Math;

/**
 * This is a singleton class responsible for setting up a VideoDataListener which
 * is what the drone forwards the camera feed to and configuring the video feed.
 * It is also responsible for managing the gimbal.
 */
public class CameraController {
    private static CameraController instance = null;
    private DJICodecManager codecManager;
    private VideoFeeder.VideoDataListener receivedVideoDataListener;
    private BaseProduct drone;
    private boolean isGimbalOverloaded = true;
    public static final String CAM_LOADED_FLAG = "camera_loaded";

    public float yawAngle;

    private CameraController(){
        //Set up a VideoDataListener
        receivedVideoDataListener = new VideoFeeder.VideoDataListener() {
            @Override
            public void onReceive(byte[] videoBuffer, int size) {
                if (codecManager != null){
                    codecManager.sendDataToDecoder(videoBuffer, size);
                }
            }
        };

        //Get a connection to the connected drone, if one exists.
        drone = DJISDKManager.getInstance().getProduct();

        Gimbal gimbal = drone.getGimbal();

        gimbal.setStateCallback(new GimbalState.Callback() {
            //Continually updates the variables below.
            @Override
            public void onUpdate(@NonNull GimbalState gimbalState) {
                isGimbalOverloaded = gimbalState.isMotorOverloaded();
                yawAngle = gimbalState.getYawRelativeToAircraftHeading();
            }
        });

    }

    /**
     * Used to access the singleton instance of the CameraController. If there is no instance,
     * one is created.
     * @return The singleton CameraController instance.
     */
    public static synchronized CameraController getInstance(){
        if (instance == null){
            instance = new CameraController();
        }
        return instance;
    }

    /**
     * Get the DJICodecManager of the DJI SDK.
     * @return The DJICodecManger
     */
    public DJICodecManager getCodecManager() {
        return codecManager;
    }

    /**
     * This method sets up the video feed to be displayed on a TextureView. It also performs
     * check that a video feed is able to be displayed. This method also performs checks that a
     * feed is available to show
     * @param cameraTextureView The TextureView on which the video should be displayed.
     * @param callingContext The context which calls this method. This should be the context that
     *                       contains the cameraTextureView
     * @param surfaceTextureListener The SurfaceTextureListener that will receive the video data
     */
    public void initPreviewer(TextureView cameraTextureView,
                              Context callingContext,
                              TextureView.SurfaceTextureListener surfaceTextureListener){
        //Make sure that the drone is connected
        if (drone == null || !drone.isConnected()){
            Toast.makeText(callingContext, R.string.disconnected, Toast.LENGTH_SHORT).show();
        } else {
            if (null != cameraTextureView){
                cameraTextureView.setSurfaceTextureListener(surfaceTextureListener);
            }
            if (!drone.getModel().equals(Model.UNKNOWN_AIRCRAFT)){
                VideoFeeder.getInstance().getPrimaryVideoFeed().addVideoDataListener(receivedVideoDataListener);
            } else{
                Toast.makeText(callingContext, "Unknown device connected. Check SDK compatibility.", Toast.LENGTH_SHORT).show();
            }
        }
    }


    /**
     * Rotates the gimbal to an absolute, i.e. fixed, rotation.
     * @param callingContext The context which calls this method.
     * @param roll Roll ration in degrees - positive is to the right
     * @param yaw Yaw rotation in degrees - positive is to the right
     * @param pitch Pitch rotation in degrees - positive is up
     * @return Returns true if gimbaling performed, false otherwise.
     */
    public boolean gimbalAbs(Context callingContext, float roll, float yaw, float pitch){

        if (isGimbalOverloaded){
            return false;
        }
        Gimbal gimbal = drone.getGimbal();
        Rotation.Builder targetRotBuilder = new Rotation.Builder().roll(roll).yaw(yaw).pitch(pitch).time(0.1).mode(RotationMode.ABSOLUTE_ANGLE);
        Rotation targetRot = targetRotBuilder.build();
        Toast.makeText(callingContext, targetRot.toString(), Toast.LENGTH_SHORT).show();
        //Rotate and check the result
        gimbal.rotate(targetRot, djiError -> {
            if (djiError != null){
                Log.e(CameraController.class.getName(), djiError.getDescription());
                Toast.makeText(callingContext, djiError.getDescription(), Toast.LENGTH_SHORT).show();
            }

        });

        return true;
    }

    /**
     * Rotates the gimbal to a relative rotation, i.e. compared to current angle.
     * @param callingContext The context which calls this method.
     * @param roll Roll ration in degrees - positive is to the right
     * @param yaw Yaw rotation in degrees - positive is to the right
     * @param pitch Pitch rotation in degrees - positive is up
     * @return Returns true if gimbaling performed, false otherwise.
     */
    public boolean gimbalRel(Context callingContext, float roll, float yaw, float pitch){

        if (Math.abs(yawAngle + yaw) > 90) {
            Log.e("gimbal","gimbal max yaw rotation reached");
            return false;
        }
        if (isGimbalOverloaded){
            return false;
        }
        Gimbal gimbal = drone.getGimbal();
        Rotation.Builder targetRotBuilder = new Rotation.Builder().roll(roll).yaw(yaw).pitch(pitch).time(0.1).mode(RotationMode.RELATIVE_ANGLE);
        Rotation targetRot = targetRotBuilder.build();
        //Toast.makeText(callingContext, targetRot.toString(), Toast.LENGTH_SHORT).show();
        //Rotate and check the result
        gimbal.rotate(targetRot, djiError -> {
            if (djiError != null){
                Log.e(CameraController.class.getName(), djiError.getDescription());
                Toast.makeText(callingContext, djiError.getDescription(), Toast.LENGTH_SHORT).show();
            }

        });

        return true;
    }

    /**
     * A special case of gimbalAbs(0, 0, -90), pointing the gimbal straight down
     * @param callingContext The context which calls this method.
     * @return Returns true if gimbaling performed, false otherwise.
     */
    public boolean gimbalDown(Context callingContext){
        return gimbalAbs(callingContext, 0, 0, -90);
    }

    /**
     * A special case of gimbalRel(0, degrees, 0), rotating to the right
     * @param callingContext The context which calls this method.
     * @param degrees The degrees of rotation
     * @return Returns true if gimbaling performed, false otherwise.
     */
    public boolean gimbalRight(Context callingContext, float degrees){
        return gimbalRel(callingContext, 0, degrees, 0);
    }

    /**
     * A special case of gimbalRel(0, -degrees, 0), rotating to the left
     * @param callingContext The context which calls this method.
     * @param degrees The degrees of rotation
     * @return Returns true if gimballing performed, false otherwise.
     */
    public boolean gimbalLeft(Context callingContext, float degrees) {
        return gimbalRel(callingContext, 0, -degrees, 0);
    }




}
