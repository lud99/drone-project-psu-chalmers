package com.dji.sdk.sample;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.view.TextureView;

import androidx.annotation.NonNull;

import dji.sdk.codec.DJICodecManager;

/**
 * This implements a TextureView.SurfaceTextureListener. The DJI SDK uses a SurfaceTexture
 * to project the video. A SurfaceTextureListener is a listener that calls methods when
 * something updates with this SurfaceTextureListener. Only Available and and Destroyed are used
 */
public class CameraSurfaceTextureListener implements TextureView.SurfaceTextureListener {
    DJICodecManager codecManager;
    Context context;
    public CameraSurfaceTextureListener(Context context){
        this.context = context;
        codecManager = CameraController.getInstance().getCodecManager();
    }


    @Override
    public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface, int width, int height) {
        if (codecManager == null) {
            codecManager = new DJICodecManager(context, surface, width, height);
        }
    }

    @Override
    public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface, int width, int height) {
        //Overridden as the interface requires this, but not actually used.
    }

    @Override
    public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface) {
        if (codecManager != null) {
            codecManager.cleanSurface();
            codecManager = null;
        }

        return false;
    }

    @Override
    public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface) {
        //Overridden as the interface requires this, but not actually used.
    }
}
