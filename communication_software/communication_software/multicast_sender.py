import socket
import json
import psutil
import threading
import time

HOST_IP = "host.docker.internal"  # Docker Desktop shortcut to host
HOST_PORT = 50000  # must match relay LISTEN_PORT


class MulticastSender:
    def __init__(self, host_ip: str, host_port: int) -> None:
        """
        host_ip: the IP assigned to your transparent network inside the container
        """
        self.host_ip = host_ip
        self.port = host_port
        self.frequency_in_seconds = 1
        self.name = "Backend1"  # TODO implement properly

        if len(self.name) > 50:
            self.name = self.name[:50]

        self.stop_event = threading.Event()

    def get_ipv4_interfaces(self):
        ips = []
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    # 127. is the loopback address
                    if not addr.address.startswith("127."):
                        ips.append(addr.address)
        return ips

    def send_packets(self):
        print(f"Start sending discovery packets for {self.host_ip}:{self.port}")

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while True:
            message = "CTH" + json.dumps(
                {
                    "msg_type": "backend_discovery",
                    "name": self.name,
                    "ip": self.host_ip,
                    "port": self.port,
                }
            )

            # To prevent UDP packets from being fragmented and risk being dropped, use a packet size in range
            # 1200 - 1400 bytes. Wifi has max size of 1500 bytes
            max_udp_packet_size = 1200

            if len(message) > max_udp_packet_size:
                print("Multicast message is larger than 1200 bytes, will not send!!!")
                continue

            s.sendto(message.encode("utf-8"), (HOST_IP, HOST_PORT))
            # print("Sent to host relay")
            time.sleep(1)

    def stop(self):
        self.stop_event.set()
