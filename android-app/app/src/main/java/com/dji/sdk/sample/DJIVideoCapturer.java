package com.dji.sdk.sample;

import org.webrtc.CapturerObserver;
import org.webrtc.NV12Buffer;
import org.webrtc.SurfaceTextureHelper;
import org.webrtc.VideoCapturer;
import org.webrtc.VideoFrame;


import android.content.Context;
import android.graphics.SurfaceTexture;
import android.media.MediaFormat;
import android.os.SystemClock;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import dji.sdk.camera.VideoFeeder;
import dji.sdk.codec.DJICodecManager;

public class DJIVideoCapturer implements VideoCapturer {
    private final static String TAG = "DJIStreamer";

    private static DJICodecManager codecManager;
    private static final ArrayList<CapturerObserver> observers = new ArrayList<CapturerObserver>();

    private final String droneDisplayName;
    private Context context;
    private CapturerObserver capturerObserver;

    public DJIVideoCapturer(String droneDisplayName){
        this.droneDisplayName = droneDisplayName;
    }

    private void setupVideoListener() {
        if (codecManager != null) {
            return; 
        }
    
   
        codecManager = new DJICodecManager(context, (SurfaceTexture) null, 0, 0);
        codecManager.enabledYuvData(true);
        codecManager.setYuvDataCallback(new DJICodecManager.YuvDataCallback() {
            @Override
           public void onYuvDataReceived(MediaFormat mediaFormat, ByteBuffer videoBuffer, int dataSize, int width, int height) {
                // 1. Calculate the size for Y and UV
                int ySize = width * height;
                int uvSize = ySize / 2; // For NV12/NV21

                // 2. Wrap the existing buffer or copy it
                // Ensure the videoBuffer position is at 0
                videoBuffer.position(0);
                
                // 3. Create the WebRTC NV12Buffer correctly
                // The second 'width' is the stride for the UV plane
                NV12Buffer buffer = new JavaI420Buffer.NV12Buffer(width, height, width, width, videoBuffer, null);

                // 4. Create the VideoFrame and send to observers
                VideoFrame videoFrame = new VideoFrame(buffer, 0, SystemClock.elapsedRealtimeNanos());
                capturerObserver.onFrameCaptured(videoFrame);
                videoFrame.release();
            }
        });
    
        // Handle video data listener for specific drone models
        addVideoDataListenerForDroneModel(this.droneDisplayName);
    }
    
    private void addVideoDataListenerForDroneModel(String droneModel) {
        switch (droneModel) {
            case "DJI Mavic Enterprise 2":
                // Only add listener once for this drone model to avoid duplication
                if (!isListenerAdded) {
                    VideoFeeder.VideoDataListener videoDataListener = new VideoFeeder.VideoDataListener() {
                        @Override
                        public void onReceive(byte[] bytes, int dataSize) {
                            // Send the encoded data to codec manager for YUV decoding
                            codecManager.sendDataToDecoder(bytes, dataSize);
                        }
                    };
                    VideoFeeder.getInstance().getPrimaryVideoFeed().addVideoDataListener(videoDataListener);
                    isListenerAdded = true; // Flag indicating that listener has been added
                }
                break;
    
            // Add more cases for different drone models as needed
            default:
                // Handle default case or add more models
                break;
        }
    }
    
    // A flag to prevent adding the listener multiple times
    private boolean isListenerAdded = false;
    

    @Override
    public void initialize(SurfaceTextureHelper surfaceTextureHelper, Context applicationContext,
                           CapturerObserver capturerObserver) {
        this.context = applicationContext;
        this.capturerObserver = capturerObserver;

        observers.add(capturerObserver);
    }

    @Override
    public void startCapture(int width, int height, int framerate) {
        // Hook onto the DJI onYuvDataReceived event
        setupVideoListener();
    }

    @Override
    public void stopCapture() throws InterruptedException {
    }

    @Override
    public void changeCaptureFormat(int width, int height, int framerate) {
        // Empty on purpose
    }

    @Override
    public void dispose() {
        // Stop receiving frames on the callback from the decoder
        if (observers.contains(capturerObserver))
            observers.remove(capturerObserver);
    }

    @Override
    public boolean isScreencast() {
        return false;
    }
}