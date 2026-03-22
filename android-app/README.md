# How the app works
Changelog
2024-02-28: Initial version

2024-03-06: Refactoring of CameraController and CameraSurfaceTextureListener

2024-03-13: Included Navigation

## Installation instructions
1. Open the project in Android Studio
2. Build the app in the Android Studio
3. Launch from Android Studios - The app is now installed and can be opened as any other app.

## Background
In order for the app to work at all and to connect to the drone, it must first be connected to the DJI SDK. This is done by registering the app using an SDK key generated on developer.dji.com. We must then connect to the drone, find the camera feed and put it on display. This app is built on a modified version of the DJI SDK Android tutorials for [registering an app](https://developer.dji.com/document/9a028db0-fcbf-421b-bd6d-feadbc60a75d) and [building a camera app](https://developer.dji.com/document/06724d27-23cf-4741-b128-fc17d2891981). However, the app has been modified as the Android standard has changed since those guides were published. What follows is a journey through the code which makes the app work. The code currently contains an RTMP livestream manager, but as this feature will not be used, it is not documented. 

## Registering the app
As previously mentioned, the DJI SDK mandates that the app is registered with DJI before the app will function. To start, the DJI SDK must be loaded into the app. This is done by calling `Helper.install()` in the overridden `attachBaseContext` method of the `mApplication` class, as written in the DJI guide. 

The actual registration process takes place in the `MainActivity` class. This activity was created by Android Studio as a template when the project was created. The `onCreate` method is called by Android when the activity is being created, i.e. when the app is starting. This method does two things. First, it calls `checkAndRequestPermissions()`, which will eventually register the app. Secondly, it does some navigation stuff to make the buttons reachable by code and enabling the navbar on the top of the screen.
### `checkAndRequestPermissions()`
The task of this function is to go through all required permissions, stored in the list `REQUIRED_PERMISSION_LIST`. It will loop through this list and check if the permission is granted. Some permissions are granted as soon as they are declared in the file `AndroidManifest.xml`. Others need to be requested from the user. If this is required, this function will request the permission. If all permissions are acquired, the function calls `startSDKRegistration()`. If they aren't, the app will show a toast and the user will need to grant the missing permissions in the Settings app.
### `startSDKRegistration()`
This function actually registers the app. This function is essentially provided by DJI. First, it checks that SDK registration isn't being attempted somewhere else by [atomically](https://en.wikipedia.org/wiki/Linearizability) checking if it already has been called. If this is not true, it will [asynchronically](https://en.wikipedia.org/wiki/Asynchrony_%28computer_programming%29) execute the registration, which frees the app to continue rendering while the registration is in progress. This is important, as registration can take upwards of 10 seconds to complete.

The actual registration takes place through the method call `DJISDKManager.getInstance().registerApp()`.  This will take the app key specified in `AndroidManifest.xml` together with some other information, such as device ID and the bundle ID of the app. The function includes an anonymous `SDKManagerCallback`, which is an interface provided by DJI to handle registration and access to the drone. When the registration is complete, `onRegsiter()` is called. If the registration is successful, the app will attempt to connect to the drone by calling `DJISDKManager.getInstance().startConnectionToProduct()` and update the UI, see `notifyStatusChange()` below. If the registration fails, this is likely because of a bad internet connection.

## Connecting to the product
This is handled essentially automatically by the SDK. There are four methods in total which are called by the SDK when there is some kind of status change. `onProductConnected()` is as the name implies called when the SDK has successfully connected to a drone. This method shows a toast and calls `notifyStatusChange()`. This behavior is repeated in `onProductDisconnected()`. Currently, there is a bug somewhere that causes the app to crash when the drone is disconnected.
 
 There is also the method `onProductChanged()`, which currently does nothing and `onComponentChanged()`. This is called when a *component*, i.e. a camera or light is attached or removed from the drone. If the new component exists, it will also call `notifyStatusChange()`. 

### `notifyStatusChange()` and `updateRunnable`
The purpose of this is to broadcast the change to all listening receivers. The connected receiver is so far only in `FirstFragment`.
### `FirstFragment` class and updating the UI
The purpose of the `FirstFragment` class is to handle the UI of the first fragment, which is the UI visible when you start the app.  When the fragment is created, the app will disable the button to view the feed and display a message to wait for app registration. When the app is registered, or when the product status changes, the app will receive a broadcast in the `BroadcastReciever receiver` and call the method `refreshSDK_UI()`.
### `refreshSDK_UI()`
This method sets the UI when there a change of status. It first checks if a product is connected by calling `DJISDKManager.getInstance().getProduct()`. If this returns `null`, there is no device connected and the UI displays an error message. If there is a connected device, the app enables the button and shows the device name connected.

## Showing the camera feed
When the user presses on the 'Show Feed'-button in the app, the app switches to the second fragment. When the UI changes to this fragment, Android calls `onCreateView()`.  This sets the `SurfaceTextureListener` of the `TextureView`, which is the "surface" where the video appears, to an instance of `CameraSurfaceTextureListener`.  `onCreateView()` also creates a `VideoFeeder.VideoDataListener()` , which is responsible for receiving the raw video data from the drone and pass it on to the decoder to display it. Finally, it does some UI stuff which was auto-generated by Android Studio. The method `onResume()` is called whenever the fragment comes into view, both on start and when it is returned to. This method calls `initPreviewer()`.

### `CameraController`
The `CameraController` is responsible for handling the access to the camera. It sets up the `receivedDataListener` and also holds the `initPreviewer()` method.

### `initPreviewer()` and displaying the video
When `initPreviewer()` is called, it starts of by checking the connection status. If no product is connected, it will display an error message. If a device is connected, it will again set the `SurfaceTextureListener` of the video surface to the instance of `CameraSurfaceTextureListener`, in case it got reassigned. If the aircraft model is known, checked by calling `product.getModel().equals(Model.UNKNOWN_AIRCRAFT))`, the previewer will tell the SDK to set the `VideoDataListener` to the one set in `onCreateView()`. 

When the `SurfaceTexture` becomes available, i.e. ready for use, a `codecManager` is created which will take the data stream received by the `VideoDataListener` and push it onto the `SurfaceTexture`. This displays the video!

### `updateVideoSize()`
By default, the `SurfaceTexture` will inflate to fit all available space. The problem is that this will mean the video will be stretched in some direction. This method fixes this by resizing the `SurfaceTexture` to have a 16:9 aspect ratio. It does this by calculating the current aspect ratio and comparing this to the desired 16:9 ratio. If the ratio is bigger, i.e. the video is too wide, the width will be reduced and if the ratio is smaller, i.e. the video is too tall, the height will be reduced. The new scaling is then applied to the `Transform` matrix of the `TextureView`.

## Navigation and flight planning
When the app is registered and the drone is connected, it is possible to control the flight plan of the drone and execute this flight plan through the app. The first step is to input the coordinates. To do this, a settings menu has been implemented. When the user taps the settings button in the menu bar, the activity `SettingsActivity` is started. This activity has four buttons, which navigate to another activity, including back to the main menu. The `CoordinatesActivity` is responsible for storing coordinates.
### The `CoordinatesActivity`
In this activity, there are a couple of text fields. In these fields, the user can input the final desired latitude, longitude and height over ground. There are also buttons for some testing locations, Gibraltarvallen and the Mossens IP soccerfield. When the user clicks apply, the `handleText()` method is called. This reads the fields, performs some sanity checks and stores them in the `FlightManager`. 

First, the functions checks that none of the fields are empty and that the altitude is greater than 5 meters. If this isn't the case, there are toasts to warn the user. After this, the method tries to grab the `FlightManager` and apply the coordinates on the public fields `input_lat`, `input_long`, and `input_alt`. If the drone is not connected, this will cause an exception which is caught and toasts.
### `FlightManager`
The flight manager is a singleton class responsible for actually communicating and managing the mission and flight plan the drone will execute. Many of the methods of the `FlightManager` are called by buttons in the `SecondFragment`, which shows the camera feed. When the user presses the "Arm flightplan", the `FlightManager.onApply()` method is called.
### `onApply()`
This method is responsible for creating the `WaypointMission` and uploading this to the drone. This mission is made up of two waypoints, the first one is 10 meters above the launch point of the drone and the second is based on the position and altitude provided in the `CoordinatesActivity`.  When the drone has navigated to the coordinate, it is set to `CONTINUE_UNTIL_END` which means the drone stays in mission mode and hovers in-place until the mission mode is aborted either through the app by tapping the "Abort" or on the controller. 

The mission is then configured in the `configWayPointMission()` method. This means that the waypoints are merged and configured with speed (6 m/s) and heading (automatic). A `WayPointAction` is configured on arrival to point the drone down. The mission is then staged to the `WaypointMissionOperator` on the drone. Finally, this is uploaded to the drone. If the drone is connected, this should be connected. 

### `startWaypointMission()`, `abortWaypointMission()`, and `endWaypointMission()`
The first method is called when the user presses the Start button in the app. This will execute the mission that is loaded and toast is no mission has been uploaded.

When the abort button is pushed, `abortWaypointMission()` is called which simply aborts the mission and puts the drone in GPS mode, enabling manual flight.

The `endWaypointMission()` is currently not in use, but is just a combination of `abortWaypointMission()` and `goHome()`, which causes the drone to go home, i.e. back to the launch point and land.

 









<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwMjk1OTkyOTNdfQ==
-->
