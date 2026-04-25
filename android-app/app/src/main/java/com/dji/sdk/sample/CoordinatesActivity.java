package com.dji.sdk.sample;

import static java.lang.Double.parseDouble;
import static java.lang.Float.parseFloat;
import static java.lang.Integer.parseInt;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

public class CoordinatesActivity extends AppCompatActivity {

    double lat;
    double lng;
    float alt;
    int jaw;

    WebsocketClientHandler websocketClientHandler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_coordinates);
        websocketClientHandler = WebsocketClientHandler.getInstance();
    }

    @Override
    protected void onResume() {
        super.onResume();
        try {
            DJIFlightManager flightManager = DJIFlightManager.getFlightManager();

            double stored_lat = flightManager.input_lat;
            double stored_lng = flightManager.input_lng;
            float stored_alt = flightManager.input_alt;

            TextView lat_out = findViewById(R.id.latOutput);
            TextView lng_out = findViewById(R.id.lonOutput);
            TextView alt_out = findViewById(R.id.altOutput);

            if (stored_alt != 0 && stored_lat != 0 && stored_lng != 0) {
                lat_out.setText(String.valueOf(stored_lat));
                lng_out.setText(String.valueOf(stored_lng));
                alt_out.setText(String.valueOf(stored_alt));
            }

        } catch (Exception ignored) {
        }
    }

    /**
     * This method converts the input in the text fields to numbers and
     * stores them in the flight manager. If something is not correct,
     * the method will show toasts to inform the user.
     * 
     * @param v The view which calls the function
     */

    public void handleText(View v) {
        EditText lat_in = findViewById(R.id.latInput);
        TextView lat_out = findViewById(R.id.latOutput);
        EditText lng_in = findViewById(R.id.lonInput);
        TextView lng_out = findViewById(R.id.lonOutput);
        EditText alt_in = findViewById(R.id.altInput);
        TextView alt_out = findViewById(R.id.altOutput);
        EditText jaw_in = findViewById(R.id.jawInput);
        TextView jaw_out = findViewById(R.id.jawOutput);

        String lat_input = lat_in.getText().toString();
        lat_out.setText(lat_input);
        String lng_input = lng_in.getText().toString();
        lng_out.setText(lng_input);
        String alt_input = alt_in.getText().toString();
        alt_out.setText(alt_input);
        String jaw_input = jaw_in.getText().toString();
        jaw_out.setText(jaw_input);

        if (jaw_input.isEmpty()) {
            jaw_input = "0";
            jaw_out.setText(jaw_input);
        } else if (parseInt(jaw_input) >= 180 || parseInt(jaw_input) <= -180) {
            Toast.makeText(this, "Bad values of jaw", Toast.LENGTH_SHORT).show();
            return;
        }

        if (lat_input.isEmpty() || lng_input.isEmpty() || alt_input.isEmpty()) {
            Toast.makeText(this, "Empty inputs not allowed", Toast.LENGTH_LONG).show();
        } else {
            try {
                lat = parseDouble(lat_input);
                lng = parseDouble(lng_input);
                alt = parseFloat(alt_input);
                jaw = parseInt(jaw_input);
            } catch (NumberFormatException e) {
                Toast.makeText(this, "All inputs must be numbers", Toast.LENGTH_LONG).show();
            }

            if (alt <= 5) {
                Toast.makeText(this, "Altitude must be higher than 5 meters, nothing loaded", Toast.LENGTH_LONG).show();
            } else {
                try {
                    DJIFlightManager.getFlightManager().input_lat = lat;
                    DJIFlightManager.getFlightManager().input_lng = lng;
                    DJIFlightManager.getFlightManager().input_alt = alt;
                    DJIFlightManager.getFlightManager().input_yaw = jaw;

                    Toast.makeText(this, "Coordinates ready for the flight manager", Toast.LENGTH_SHORT).show();
                    Log.d("info", "coordinates hopefully loaded");

                } catch (Exception e) {
                    Toast.makeText(this, "Coordinates failed to reach Flightmanager, check if drone is connected",
                            Toast.LENGTH_LONG).show();
                }
            }
        }
    }

    /**
     * Returns from the Coordinates activity to the main screen.
     * 
     * @param v The calling view
     */
    public void backToHome(View v) {
        Intent openMainActivity = new Intent(CoordinatesActivity.this, MainActivity.class);
        openMainActivity.addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT);
        startActivity(openMainActivity);
    }

    /**
     * If there is an active connection, managed in the Server Setting menu,
     * this button can load the coordinates from the server.
     * To prevent the app from crashing if this fails, the check is performed in an
     * AsyncTask.
     * WARNING! The server must output in the correct format, and no check is made
     * for that
     */
    public void loadCoordsFromServer(View v) {
        String message ="{\"msg_type\": \"Coordinate_request\"}";
        if (WebsocketClientHandler.isInstanceCreated() && websocketClientHandler.send(message)) {
            AsyncTask.execute(getAndApplyCoordinateRunnable);

        } else {
            Toast.makeText(this, "No server connected...", Toast.LENGTH_SHORT).show();
        }

    }

    /**
     * Loads the coordinates of Mossens IP (57.684478 / 11.979772). Useful for
     * debugging if
     * you are at Chalmers.
     * 
     * @param v The calling View
     */
    public void loadMossCoords(View v) {

        EditText lat_in = findViewById(R.id.latInput);
        EditText lon_in = findViewById(R.id.lonInput);
        EditText alt_in = findViewById(R.id.altInput);
        EditText jaw_in = findViewById(R.id.jawInput);
        lat_in.setText("40.783629844534545");
        lon_in.setText("-77.85258263534078");
        alt_in.setText("45");
        jaw_in.setText("0");
    }

    /**
     * This Runnable gets the string and and sends it to the UI thread.
     */
    final private Runnable getAndApplyCoordinateRunnable = new Runnable() {
        @Override
        public void run() {
            String lastString = websocketClientHandler.getLastStringReceived();
            if (lastString.contains("{")) {
                runOnUiThread(applyCoordinatesOnUI);
            }
        }
    };

    /**
     * This Runnable applies the coordinates to the field.
     */
    final private Runnable applyCoordinatesOnUI = new Runnable() {
        @Override
        public void run() {
            String lastString = websocketClientHandler.getLastStringReceived();

            try {
                JSONObject json = new JSONObject(lastString);

                String lat = json.getString("lat");
                String lng = json.getString("lng");
                String alt = json.getString("alt");
                String angle = json.getString("angle");

                EditText lat_in = findViewById(R.id.latInput);
                EditText lng_in = findViewById(R.id.lonInput);
                EditText alt_in = findViewById(R.id.altInput);
                EditText jaw_in = findViewById(R.id.jawInput);

                lat_in.setText(lat);
                lng_in.setText(lng);
                alt_in.setText(alt);
                jaw_in.setText(angle);

            } catch (Exception e) {
                e.printStackTrace(); // Handle parsing error
            }

        }
    };

}