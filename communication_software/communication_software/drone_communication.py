import time
import json
from typing import Optional
import redis.exceptions
import websockets
from websockets import WebSocketServerProtocol

import threading
import asyncio
import redis
import os
import cv2
import numpy as np
from communication_software.multicast_sender import MulticastSender

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.sdp import candidate_from_sdp
import communication_software.common.json_schemas as json_schemas


try:
    r = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    r.ping()
    r.flushdb()  # Removes all stuff, as stopping docker containers is not enough to clear it
    print("Successfully connected to Redis (Drone Communication Server)!")
except redis.exceptions.ConnectionError as e:
    print(f"Error connecting to Redis (Drone Communication Server): {e}")
    exit()

COMMAND_CHANNEL = "drone_commands"


from aiortc import RTCConfiguration, RTCIceServer  # noqa: E402

ice_configuration = RTCConfiguration(
    iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
)


class DroneCommunication:
    def __init__(self) -> None:
        self.connections = {}  # Active WebSocket connections
        self.connection_counter = 0
        # self.drone_coordinates = []  # List of drone coordinates

        self.loop = None
        self.redis_listener_stop_event = threading.Event()
        self.redis_listener_task = None
        self.peer_connections = {}

        # For testing
        with open(
            "./src/communication_software/communication_software/common/test_mock_mission.json",
            "r",
        ) as f:
            self.task_list = json.loads(f.read())
        self.task_index = 0

    async def start_websocket_server(self, ip: str) -> None:
        """Starts WebSocket server."""
        print(f"Initializing WebSocket on IP: {ip}")

        if not self.loop:
            print("ERROR: Event loop not set before starting WebSocket server!")
            try:
                self.loop = asyncio.get_running_loop()
                print(
                    "Fallback: Acquired event loop within send_coordinates_websocket."
                )
            except RuntimeError:
                print("FATAL: Could not acquire event loop.")
                return  # Exit if loop cannot be found

        if not self.redis_listener_task or not self.redis_listener_task.is_alive():
            print(
                "Warning: Redis listener thread not started or alive. Starting it now."
            )
            self.start_redis_listener_thread()  # Assumes self.loop is set

        websocket_port = int(os.environ.get("WEBSOCKET_PORT", 14500))
        host_ip = str(os.environ.get("HOST_IP"))

        server = await websockets.serve(self.webs_server, ip, websocket_port)
        print(f"WebSocket server started on ws://{ip}:{websocket_port}")

        # Start multicasting now after websocket server has started
        sender = MulticastSender(host_ip, websocket_port)
        multicast_thread = threading.Thread(target=sender.send_packets)
        multicast_thread.start()

        try:
            await server.wait_closed()
        finally:
            print("WebSocket server stopping...")
            self.redis_listener_stop_event.set()
            if self.redis_listener_task and self.redis_listener_task.is_alive():
                print("Waiting for Redis listener thread to finish...")
                self.redis_listener_task.join(timeout=5)  # Wait briefly for cleanup
                if self.redis_listener_task.is_alive():
                    print("Warning: Redis listener thread did not stop gracefully.")
            print("WebSocket server stopped.")
            sender.stop()
            multicast_thread.join()
            print("Stopped multicast sender thread")

    # def transform_coordinates(self, coordinates: Coordinate, angle: int) -> tuple:
    #     """Transforms coordinates into required format."""
    #     lat = str(coordinates.lat)[:9]
    #     lng = str(coordinates.lng)[:9]
    #     alt = str(coordinates.alt)[:3]
    #     new_angle = str(angle)
    #     return (lat, lng, alt, new_angle)

    def redis_command_listener(self, redis_client, channel, stop_event):
        """Listens for messages on the specified Redis channel in a blocking loop."""
        print(f"[REDIS THREAD] Listener thread started for channel '{channel}'.")
        pubsub = None
        listener_redis_conn = None

        # Ensure the event loop reference is available
        if not self.loop:
            print(
                "[REDIS THREAD] FATAL: Event loop not available. Listener thread cannot proceed."
            )
            return  # Exit thread if loop isn't set

        while not stop_event.is_set():  # Loop until stop event is set
            try:
                listener_redis_conn = redis.Redis(
                    host=redis_client.connection_pool.connection_kwargs.get(
                        "host", "redis"
                    ),  # Use 'redis' as default host
                    port=redis_client.connection_pool.connection_kwargs.get(
                        "port", 6379
                    ),
                    db=redis_client.connection_pool.connection_kwargs.get("db", 0),
                    decode_responses=True,
                    socket_connect_timeout=5,  # Add timeout
                    socket_keepalive=True,  # Add keepalive
                )
                listener_redis_conn.ping()  # Verify connection
                pubsub = listener_redis_conn.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(channel)
                print(
                    f"[REDIS THREAD] Subscribed successfully to '{channel}'. Waiting..."
                )

                for message in pubsub.listen():
                    if stop_event.is_set():
                        print(
                            "[REDIS THREAD] Stop event detected, exiting listen loop."
                        )
                        break  # Exit inner listen loop

                    if message and message.get("type") == "message":
                        print(f"[REDIS THREAD] Received message: {message['data']}")
                        coro = self.process_redis_command(message["data"])
                        asyncio.run_coroutine_threadsafe(coro, self.loop)

                        try:
                            pass  # Fire-and-forget is usually fine here
                        except Exception as e:
                            print(
                                f"[REDIS THREAD] Error submitting/executing command: {e}"
                            )
                    else:
                        pass

                if not stop_event.is_set():
                    print(
                        "[REDIS THREAD] Pubsub listen loop finished unexpectedly. Will attempt reconnect."
                    )
                    # Force cleanup before retry delay
                    if pubsub:
                        pubsub.close()
                    if listener_redis_conn:
                        listener_redis_conn.close()
                    pubsub = None
                    listener_redis_conn = None
                    time.sleep(5)  # Wait before attempting to reconnect
                    continue  # Continue the outer while loop to retry connection

            except redis.exceptions.ConnectionError as e:
                print(f"[REDIS THREAD] Connection error: {e}. Retrying in 5 seconds...")
                if pubsub:
                    pubsub.close()  # Use close() for pubsub
                if listener_redis_conn:
                    listener_redis_conn.close()  # Use close() for connection
                pubsub = None
                listener_redis_conn = None
                time.sleep(5)  # Wait before attempting to reconnect
            except redis.exceptions.TimeoutError as e:
                print(
                    f"[REDIS THREAD] Redis command timeout: {e}. Retrying in 5 seconds..."
                )
                if pubsub:
                    pubsub.close()
                if listener_redis_conn:
                    listener_redis_conn.close()
                pubsub = None
                listener_redis_conn = None
                time.sleep(5)
            except Exception as e:
                # Log unexpected errors more informatively
                import traceback

                print(f"[REDIS THREAD] Unexpected error in listener loop: {e}")
                print(traceback.format_exc())
                break  # Exit outer loop on unexpected errors for safety now
            finally:
                if pubsub:
                    try:
                        pubsub.unsubscribe(channel)
                    except Exception as _e:
                        pass  # Ignore errors during cleanup
                    try:
                        pubsub.close()
                    except Exception as _e:
                        pass
                if listener_redis_conn:
                    try:
                        listener_redis_conn.close()
                    except Exception as _e:
                        pass

        print("[REDIS THREAD] Listener thread finished.")

    async def process_redis_command(self, redis_message_data):
        """Processes a command received from Redis (runs in the main event loop)."""
        print(
            f"\n[PROCESS CMD] Processing command received from Redis. Data type: {type(redis_message_data)}"
        )
        try:
            if isinstance(redis_message_data, bytes):
                redis_message_data = redis_message_data.decode("utf-8")

            print(f"[PROCESS CMD] Raw Command Data: {redis_message_data}")

            message = json_schemas.parse_drone_message(redis_message_data)
            if not isinstance(message, json_schemas.BackendToDroneMessage):
                print(
                    "f[PROCESS CMD] Message f{message.msg_type} is not valid to send to drones"
                )
                return

            connection_id = message.drone_id

            print(
                f"[PROCESS CMD] Parsed: Drone='{connection_id}', msg_type='{message.msg_type}'"
            )

            print(
                f"[PROCESS CMD] Current active connections: {len(self.connections.items())}"
            )

            if connection_id not in self.connections:
                print(f"[PROCESS CMD] {connection_id} is not connected!!")
                return

            connection_ws = self.connections[connection_id]

            if not connection_ws:
                print(
                    f"[PROCESS CMD] ERROR: WebSocket connection {connection_id} not open or not found."
                )
                if (
                    connection_id in self.connections
                ):  # Check if it still exists in dict
                    self.cleanup_connection(connection_id)
                    return

            try:
                print(
                    f"Sending message to WebSocket {connection_id}: {redis_message_data}"
                )
                await connection_ws.send(redis_message_data)
            except websockets.exceptions.ConnectionClosed:
                print(
                    f"[PROCESS CMD] ERROR: WebSocket connection {connection_id} closed before sending."
                )
                self.cleanup_connection(connection_id)  # Clean up if closed
            except Exception as send_err:
                print(
                    f"[PROCESS CMD] ERROR: Failed to send message to WebSocket {connection_id}: {send_err}"
                )

        except Exception as e:
            # Log unexpected errors more informatively
            import traceback

            print(f"[PROCESS CMD] ERROR: Unexpected error processing command: {e}")
            print(traceback.format_exc())

    async def webs_server(self, ws: WebSocketServerProtocol) -> None:
        """Handles WebSocket connections."""
        connection_id = "NON_droneid_" + str(self.connection_counter)
        self.connection_counter += 1
        print(f"Client connected. Assigned to temporary id {connection_id}")

        self.connections[connection_id] = ws

        try:
            while True:
                data = await ws.recv()
                print(f"Received from {connection_id}: {data}")
                new_connection_id = await self.on_message(data, connection_id, ws)
                if new_connection_id is not None:
                    connection_id = new_connection_id
        except websockets.exceptions.ConnectionClosedOK as e:
            print(f"Client {connection_id} closed (cleanly):", e.code)
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"ConnectionClosedError: {e.code}, {e.reason}")
            print(f"Client {connection_id} disconnected.")
        finally:
            self.cleanup_connection(connection_id)

    async def on_message(
        self, frame: str, connection_id: str, ws: WebSocketServerProtocol
    ) -> Optional[str]:
        """Processes incoming messages."""
        try:
            message = json_schemas.parse_drone_message(frame)
            # print(f"Received message: {data}")

            # Route messages based on `msg_type`

            if isinstance(message, json_schemas.DebugMessage):
                print(f"Debug message from {connection_id}: {message.message}")

            elif isinstance(message, json_schemas.TelemetryMessage):
                self.handle_telemetry_message(message, connection_id)

            elif isinstance(message, json_schemas.DroneRegistrationMessage):
                return await self.handle_registration_message(
                    message, connection_id, ws
                )

            elif isinstance(message, json_schemas.TaskEventMessage):
                await self.handle_task_event_message(message, connection_id)

            # WebRTC
            elif isinstance(message, json_schemas.WebRTCCandidateMessage):
                await self.handle_webrct_candidate_message(message, connection_id)

            elif isinstance(message, json_schemas.WebRTCAnswerMessage):
                await self.handle_webrct_answer_message(message, connection_id)

            else:
                print(f"Unhandled `msg_type`: {message.msg_type}")

        except Exception as e:
            print(f"Error processing message from {connection_id}: {frame} {e}")

        return None

    # async def send_coords(self, connection_id: str) -> None:
    #     """Sends assigned coordinates to the client."""
    #     if connection_id in self.coordinates:
    #         lat, lng, alt, angle = self.coordinates[connection_id]
    #         message = {
    #             "msg_type": "Coordinate_request",
    #             "lat": lat,
    #             "lng": lng,
    #             "alt": alt,
    #             "angle": angle,
    #         }
    #         try:
    #             await self.connections[connection_id].send(json.dumps(message))
    #             print(f"Sent coordinates to client {connection_id}: {message}")
    #         except websockets.exceptions.ConnectionClosed:
    #             print(f"Connection {connection_id} closed, cleaning up.")
    #             self.cleanup_connection(connection_id)
    #     else:
    #         print(f"No coordinates found for {connection_id}")

    def cleanup_connection(self, connection_id: str) -> None:
        """Cleans up connections and PeerConnections when a client disconnects."""
        self.connections.pop(connection_id, None)

        r.delete(f"telemetry_drone{connection_id}")
        r.delete(f"capabilities_drone{connection_id}")

        print(f"Connection {connection_id} removed.")

    def start_redis_listener_thread(self):
        """Starts the Redis listener thread."""
        if not self.loop:
            print("ERROR: Cannot start Redis listener thread, event loop is not set.")
            return

        if self.redis_listener_task and self.redis_listener_task.is_alive():
            print("Redis listener thread already running.")
            return

        print("Starting Redis listener thread...")
        self.redis_listener_stop_event.clear()  # Ensure stop event is clear
        self.redis_listener_task = threading.Thread(
            target=self.redis_command_listener,
            # Pass the global 'r' redis client, channel name, and stop event
            args=(
                r,
                COMMAND_CHANNEL,
                self.redis_listener_stop_event,
            ),  # Removed 'self'/'instance' from args
            daemon=True,  # Thread will exit if main program exits
        )
        self.redis_listener_task.start()
        print("Started Redis listener thread.")

    def handle_telemetry_message(
        self, message: json_schemas.TelemetryMessage, connection_id
    ):
        # For debugginig
        # lat = data.get("latitude")
        # long = data.get("longitude")
        # altitude = data.get("altitude")
        # print(f"Handling position: lat={lat}, long={long}, altitude={altitude}")
        try:
            r.set(
                f"telemetry_drone{connection_id}",
                message.telemetry.model_dump_json(),
                ex=60,
            )

        except (TypeError, redis.exceptions.RedisError) as e:
            print(
                f"Error processing telemetry message {message.model_dump_json()}: {e}"
            )

    async def handle_registration_message(
        self,
        message: json_schemas.DroneRegistrationMessage,
        connection_id: str,
        ws: WebSocketServerProtocol,
    ) -> Optional[str]:
        try:
            if message.drone_id in self.connections:
                print(f"Drone id {message.drone_id} is already connected or used!!")
                await ws.close()
                return None

            # Move to proper drone id
            self.connections.pop(connection_id)
            self.connections[message.drone_id] = ws

            print(f"Moved websocket id from {connection_id} -> {message.drone_id}")

            r.set(
                f"capabilities_drone{message.drone_id}",
                message.capabilities.model_dump_json(),
            )
            r.set(
                f"telemetry_drone{message.drone_id}",
                message.telemetry.model_dump_json(),
                ex=60,
            )

            if message.capabilities.camera is not None:
                self.create_peer_connection(message.drone_id)
                await self.start_drone_stream(message.drone_id)

            # Test sending a task
            print(f"Sending task {self.task_index}")
            await ws.send(json.dumps(self.task_list[self.task_index]))
            self.task_index += 1

            return message.drone_id

        except (TypeError, redis.exceptions.RedisError) as e:
            print(
                f"Error processing registration message {message.model_dump_json()}: {e}"
            )

        return None

    async def handle_task_event_message(
        self, message: json_schemas.TaskEventMessage, connection_id
    ):
        if self.task_index < len(self.task_list):
            print(f"Sending task {self.task_index}")
            await self.connections[connection_id].send(
                json.dumps(self.task_list[self.task_index])
            )
            self.task_index += 1

    async def handle_webrct_candidate_message(
        self, message: json_schemas.WebRTCCandidateMessage, connection_id
    ):
        candidate_sdp = message.candidate

        sdp_mid = message.id
        mline_index = message.label

        # Parse the candidate line properly
        rtc_candidate = candidate_from_sdp(candidate_sdp)

        rtc_candidate = RTCIceCandidate(
            foundation=rtc_candidate.foundation,
            component=rtc_candidate.component,
            priority=rtc_candidate.priority,
            ip=rtc_candidate.ip,
            port=rtc_candidate.port,
            protocol=rtc_candidate.protocol,
            type=rtc_candidate.type,
            sdpMid=sdp_mid,
            sdpMLineIndex=mline_index,
            relatedAddress=rtc_candidate.relatedAddress,
            relatedPort=rtc_candidate.relatedPort,
            tcpType=rtc_candidate.tcpType,
        )

        await self.peer_connections[connection_id].addIceCandidate(rtc_candidate)
        print(f"[RTC] Added ICE candidate from {connection_id}: {candidate_sdp}")

    async def handle_webrct_answer_message(
        self, message: json_schemas.WebRTCAnswerMessage, connection_id
    ):
        if connection_id in self.peer_connections:
            sdp = message.sdp
            sdp_type = message.msg_type

            if sdp_type not in ("answer", "offer"):
                print(f"[RTC ERROR] Unexpected SDP type: {sdp_type}")
                return

            await self.peer_connections[connection_id].setRemoteDescription(
                RTCSessionDescription(sdp=sdp, type=sdp_type)
            )

        else:
            print(
                f"[DroneStream] ERROR: Peer connection for {connection_id} not found."
            )
        print(f"Received SDP answer from {connection_id}: {message}")

    ###WEBBRTC###

    async def send_message(self, connection_id, message):
        """ "Sends a message to the WebSocket connection."""
        print(
            f"[DroneStream] Sending message: {message} to connection ID: {connection_id}"
        )
        try:
            await self.connections[connection_id].send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection {connection_id} closed, cleaning up.")
            self.cleanup_connection(connection_id)

    async def start_drone_stream(self, connection_id):
        """Initiates the WebRTC stream with the drone."""
        try:
            offer = await self.peer_connections[connection_id].createOffer()
            print(f"[DroneStream] WebRTC offer created: {offer.sdp}")
            await self.peer_connections[connection_id].setLocalDescription(offer)
            await self.send_message(
                connection_id, {"msg_type": "offer", "sdp": offer.sdp}
            )

        except Exception as e:
            print(f"[DroneStream] Error in createOffer(): {e}")

    def create_peer_connection(self, connection_id):
        """Create and configure the RTCPeerConnection."""
        try:
            self.peer_connections[connection_id] = RTCPeerConnection(
                configuration=ice_configuration
            )

            @self.peer_connections[connection_id].on("icecandidate")
            async def on_ice_candidate(event):
                if event.candidate:
                    await self.send_message(
                        connection_id,
                        {
                            "msg_type": "candidate",
                            "candidate": event.candidate.to_sdp(),
                        },
                    )
                else:
                    print("[DroneStream] End of ICE candidates")

            @self.peer_connections[connection_id].on("track")
            def on_track(track):

                print(f"[DroneStream] Received track: {track.kind}")

                async def process_video(track):
                    while True:
                        try:
                            frame = await track.recv()  # receives yuv420p frame
                            yuv_frame = frame.to_ndarray(
                                format="yuv420p"
                            )  # Convert to YUV420p

                            await self.set_frame(connection_id, yuv_frame)

                        except Exception as e:
                            print(
                                f"[DroneStream] Error processing video frame for connection {connection_id}: {e}"
                            )
                            break

                if track.kind == "video":
                    asyncio.create_task(process_video(track))

            @self.peer_connections[connection_id].on("connectionstatechange")
            async def on_connection_state_change():
                state = self.peer_connections[connection_id].connectionState
                states = {
                    "new": "Connecting…",
                    "checking": "Checking connection…",
                    "connected": "Online",
                    "disconnected": "Disconnecting…",
                    "closed": "Offline",
                    "failed": "Error",
                }
                print(f"[DroneStream] State: {states.get(state, 'Unknown')}")

            # Add transceivers for video and audio
            self.peer_connections[connection_id].addTransceiver(
                "video", direction="recvonly"
            )
            self.peer_connections[connection_id].addTransceiver(
                "audio", direction="recvonly"
            )

            print(
                f"[DroneStream] Created RTCPeerConnection: {self.peer_connections[connection_id]}"
            )
        except Exception as e:
            print(f"[DroneStream] Failed to create PeerConnection: {e}")

    # async def handle_incoming_webrtc_msg(self, connection_id, message):
    #     """Route incoming WebRTC messages to the appropriate DroneStream."""
    #     try:
    #         await self.on_message(message, connection_id)
    #         print(
    #             f"[Stream Manager] Message routed to DroneStream ({connection_id}) successfully."
    #         )
    #     except KeyError as e:
    #         print(f"[Stream Manager] Error: {e}")

    ##THIS IS THE FUNCTION THAT HANDLES THE VIDEO STREAM##
    async def set_frame(self, connection_id: str, img: np.ndarray):
        try:
            # Convert the image to a buffer (JPEG format)
            ret, buffer = cv2.imencode(".jpg", img)
            if ret:
                frame_str = buffer.tobytes().decode("latin1")

                # Redis pipeline for storing the frame and setting TTL
                drone_number = await self.get_connection_id_number(connection_id)
                redis_key = f"frame_drone{drone_number}"
                with r.pipeline() as pipe:
                    pipe.set(redis_key, frame_str)  # Save the frame
                    pipe.expire(redis_key, 60)  # Set expiration (60 seconds)
                    pipe.execute()  # Execute both commands together

            else:
                print(f"Failed to encode frame for connection {connection_id}")

        except Exception as e:
            print(f"Error in set_frame: {e}")

    async def get_connection_id_number(self, connection_id):
        key_list = list(self.connections.keys())
        connection_id_number = key_list.index(connection_id) + 1
        return connection_id_number
