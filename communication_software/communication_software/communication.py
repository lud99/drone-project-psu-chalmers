import time
import json
import redis.exceptions
import websockets
from websockets import WebSocketServerProtocol
from communication_software.convex_hull_scalable import (
    Coordinate,
)
import threading
import asyncio
import redis
import os
import cv2
import numpy as np
from communication_software.multicast_sender import MulticastSender

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.sdp import candidate_from_sdp


try:
    r = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    r.ping()
    print("Successfully connected to Redis (Communication Server)!")
except redis.exceptions.ConnectionError as e:
    print(f"Error connecting to Redis (Communication Server): {e}")
    exit()

COMMAND_CHANNEL = "drone_commands"


from aiortc import RTCConfiguration, RTCIceServer  # noqa: E402

ice_configuration = RTCConfiguration(
    iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
)


class Communication:
    def __init__(self) -> None:
        self.connections = {}  # Active WebSocket connections
        self.coordinates = {}  # Coordinates for each client
        self.drone_coordinates = []  # List of drone coordinates
        self.client_index = 0  # Tracks which coordinate to assign next

        self.loop = None
        self.redis_listener_stop_event = threading.Event()
        self.redis_listener_task = None
        self.peer_connections = {}

    async def send_coordinates_websocket(
        self, ip: str, drone_origins: list, angles: list
    ) -> None:
        """Starts WebSocket server and initializes drone coordinates."""
        print(f"Initializing WebSocket on IP: {ip}")
        self.drone_coordinates = [
            (self.transform_coordinates(coord, angle))
            for coord, angle in zip(drone_origins, angles)
        ]
        print(f"Prepared drone coordinates: {self.drone_coordinates}")

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

    def transform_coordinates(self, coordinates: Coordinate, angle: int) -> tuple:
        """Transforms coordinates into required format."""
        lat = str(coordinates.lat)[:9]
        lng = str(coordinates.lng)[:9]
        alt = str(coordinates.alt)[:3]
        new_angle = str(angle)
        return (lat, lng, alt, new_angle)

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

    async def process_redis_command(self, message_data):
        """Processes a command received from Redis (runs in the main event loop)."""
        print(
            f"\n[PROCESS CMD] Processing command received from Redis. Data type: {type(message_data)}"
        )
        try:
            if isinstance(message_data, bytes):
                message_data = message_data.decode("utf-8")

            print(f"[PROCESS CMD] Raw Command Data: {message_data}")
            data = json.loads(message_data)

            target_drone_id_str = data.get("target_drone_id")
            command = data.get("command")
            payload = data.get("payload", {})  # Default to empty dict if missing
            timestamp = data.get("timestamp")

            print(
                f"[PROCESS CMD] Parsed: Drone='{target_drone_id_str}', Cmd='{command}', Payload={payload}, TS={timestamp}"
            )

            if target_drone_id_str is None or command is None:
                print(
                    f"[PROCESS CMD] ERROR: Missing 'target_drone_id' or 'command' in message: {data}"
                )
                return

            try:
                target_drone_id = int(target_drone_id_str)
            except ValueError:
                print(
                    f"[PROCESS CMD] ERROR: Invalid 'target_drone_id': {target_drone_id_str}. Must be an integer."
                )
                return

            response = {
                "msg_type": command,
                **payload,  # Merge payload into the response if needed, or handle separately
            }
            response_json = json.dumps(response)

            active_connections = list(
                self.connections.items()
            )  # Get list of (id, ws) pairs
            print(
                f"[PROCESS CMD] Current active connections: {len(active_connections)}"
            )
            connection_index = target_drone_id - 1

            if 0 <= connection_index < len(active_connections):
                connection_id, connection_ws = active_connections[connection_index]
                print(
                    f"[PROCESS CMD] Target Index: {connection_index}, Connection ID: {connection_id}"
                )

                if connection_ws:
                    try:
                        print(
                            f"Sending message to WebSocket {connection_id}: {response_json}"
                        )
                        await connection_ws.send(response_json)
                        print(
                            f"[PROCESS CMD] Successfully sent command '{command}' to drone {target_drone_id} (WS: {connection_id})."
                        )
                    except websockets.exceptions.ConnectionClosed:
                        print(
                            f"[PROCESS CMD] ERROR: WebSocket connection {connection_id} closed before sending."
                        )
                        self.cleanup_connection(connection_id)  # Clean up if closed
                    except Exception as send_err:
                        print(
                            f"[PROCESS CMD] ERROR: Failed to send message to WebSocket {connection_id}: {send_err}"
                        )
                else:
                    print(
                        f"[PROCESS CMD] ERROR: WebSocket connection {connection_id} not open or not found."
                    )
                    if (
                        connection_id in self.connections
                    ):  # Check if it still exists in dict
                        self.cleanup_connection(connection_id)

            else:
                print(
                    f"[PROCESS CMD] ERROR: Target drone ID {target_drone_id} (index {connection_index}) is out of range for available connections ({len(active_connections)})."
                )

        except json.JSONDecodeError:
            print(
                f"[PROCESS CMD] ERROR: Failed to decode JSON from Redis message: {message_data}"
            )
        except Exception as e:
            # Log unexpected errors more informatively
            import traceback

            print(f"[PROCESS CMD] ERROR: Unexpected error processing command: {e}")
            print(traceback.format_exc())

    async def webs_server(self, ws: WebSocketServerProtocol) -> None:
        """Handles WebSocket connections."""
        print("Client connected.")
        connection_id = str(id(ws))
        self.connections[connection_id] = ws
        self.create_peer_connection(connection_id)
        # await self.start_drone_stream(connection_id)

        available_coords = [
            coord
            for coord in self.drone_coordinates
            if coord not in self.coordinates.values()
        ]
        assigned_coord = (
            available_coords[0]
            if available_coords
            else self.drone_coordinates[self.client_index % len(self.drone_coordinates)]
        )
        self.coordinates[connection_id] = assigned_coord
        self.client_index += 1
        print(f"Assigned coordinate {assigned_coord} to client {connection_id}")

        try:
            while True:
                data = await ws.recv()
                print(f"Received from {connection_id}: {data}")
                await self.on_message(data, connection_id)
        except websockets.exceptions.ConnectionClosedOK as e:
            print(f"Client {connection_id} closed (cleanly):", e.code)
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"ConnectionClosedError: {e.code}, {e.reason}")
            print(f"Client {connection_id} disconnected.")
        finally:
            self.cleanup_connection(connection_id)

    async def on_message(self, frame: str, connection_id: str) -> None:
        """Processes incoming messages."""
        try:
            data = json.loads(frame)
            # print(f"Received message: {data}")

            msg_type = data.get("msg_type")
            if not msg_type:
                raise ValueError(f"Missing `msg_type` in message: {data}")

            # Route messages based on `msg_type`
            if msg_type == "Coordinate_request":
                await self.send_coords(connection_id)
            elif msg_type == "Position":
                self.incoming_position_handler(data, connection_id)
            elif msg_type == "Debug":
                msg = data.get("msg", "")
                print(f"Debug message: {msg}")
            elif msg_type == "candidate":
                candidate_sdp = data.get("candidate")
                sdp_mid = data.get("sdpMid", data.get("id", "0"))
                mline_index = int(data.get("sdpMLineIndex", data.get("label", 0)))

                if not candidate_sdp:
                    print(f"[RTC WARNING] Missing 'candidate' field: {data}")
                    return

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

                await self.peer_connections[connection_id].addIceCandidate(
                    rtc_candidate
                )
                print(
                    f"[RTC] Added ICE candidate from {connection_id}: {candidate_sdp}"
                )

            elif msg_type == "answer":
                # Todo: Handle SDP answer
                if connection_id in self.peer_connections:
                    sdp = data.get("sdp")
                    sdp_type = data.get("type") or data.get("msg_type")

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
                print(f"Received SDP answer from {connection_id}: {data}")
            else:
                print(f"Unhandled `msg_type`: {msg_type}")
        except json.JSONDecodeError:
            print(f"Malformed JSON received from {connection_id}: {frame}")
        except Exception as e:
            print(f"Error processing message from {connection_id}: {e}")

    async def send_coords(self, connection_id: str) -> None:
        """Sends assigned coordinates to the client."""
        if connection_id in self.coordinates:
            lat, lng, alt, angle = self.coordinates[connection_id]
            message = {
                "msg_type": "Coordinate_request",
                "lat": lat,
                "lng": lng,
                "alt": alt,
                "angle": angle,
            }
            try:
                await self.connections[connection_id].send(json.dumps(message))
                print(f"Sent coordinates to client {connection_id}: {message}")
            except websockets.exceptions.ConnectionClosed:
                print(f"Connection {connection_id} closed, cleaning up.")
                self.cleanup_connection(connection_id)
        else:
            print(f"No coordinates found for {connection_id}")

    def cleanup_connection(self, connection_id: str) -> None:
        """Cleans up connections and PeerConnections when a client disconnects."""
        self.connections.pop(connection_id, None)
        self.coordinates.pop(connection_id, None)
        # self.peer_connections.pop(connection_id, None)
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

    def incoming_position_handler(self, data, connection_id):
        """Handles incoming position data."""
        # For debugginig
        # lat = data.get("latitude")
        # long = data.get("longitude")
        # altitude = data.get("altitude")
        # print(f"Handling position: lat={lat}, long={long}, altitude={altitude}")
        try:
            json_data_string = json.dumps(data)
            r.set(f"position_drone{connection_id}", json_data_string, ex=10)
        except (TypeError, redis.exceptions.RedisError) as e:
            print(f"Error processing position data: {e}")

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

    async def handle_incoming_webrtc_msg(self, connection_id, message):
        """Route incoming WebRTC messages to the appropriate DroneStream."""
        try:
            await self.on_message(message, connection_id)
            print(
                f"[Stream Manager] Message routed to DroneStream ({connection_id}) successfully."
            )
        except KeyError as e:
            print(f"[Stream Manager] Error: {e}")

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
