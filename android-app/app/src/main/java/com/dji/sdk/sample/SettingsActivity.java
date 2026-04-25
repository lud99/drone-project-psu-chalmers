package com.dji.sdk.sample;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

public class SettingsActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);
    }

    /**
     * launches the setting tab
     * @param v The calling view
     */
    public void openCoords(View v) {
        Intent coordsIntent = new Intent(SettingsActivity.this, CoordinatesActivity.class);
        startActivity(coordsIntent);
    }

    /**
     * launches the Server settings tab
     * @param v The calling view
     */
    public void openServer(View v) {
        //launches the setting tab
        Intent serverIntent = new Intent(SettingsActivity.this, ServerActivity.class);
        startActivity(serverIntent);
    }

    /**
     * launches the Server settings tab
     * @param v The calling view
     */
    public void openGimbalSetting(View v) {
        //launches the setting tab
        //Intent gimbalIntent = new Intent(SettingsActivity.this.getActivity(), ServerActivity.class);
        //startActivity(gimbalIntent);
        Toast.makeText(v.getContext(), "Not implemented", Toast.LENGTH_SHORT).show();
    }

    /**
     * Returns to the main menu
     * @param v The calling view
     */
    public void backToHome(View v){
        Intent openMainActivity = new Intent(SettingsActivity.this, MainActivity.class);
        openMainActivity.addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT);
        startActivity(openMainActivity);
    }

}
