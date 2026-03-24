# Mission planning
Add more specific description of how the system works when everything is implemented and works

# Creating new mission types
More mission types can be created in the module: ```missions.py```.
They should inherit from the abstract class ```Mission```.

# Adding audio files to DJI
To add a new audio file, add them in the file ```audio_file_mapping.json``` in the appropriate category.
To actually upload the audio file to a DJI drone do the following:
    1. Upload a mp3 file to the android Phone with the same file name as in the json file.
    2. Attach the speaker to the DJI drone.
    3. Connect the Android phone to the controller and connect the controller to the drone.
    4. Go into the App: DJI Pilot
    5. Click the three dots ... in the upper right corner
    6. Click the speaker icon
    7. Click upload file and select the correct file.

# Testing Mission Planning
The mission planning can be tested using the file: ```test_auto_mission_suggest.py``` .
This simulates simulated drones, detected objects and registered watch-areas.
More documentation can be found in the file.