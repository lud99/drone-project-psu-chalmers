from socket import *
import json
import psutil
import threading
import time

class MulticastSender:
    def __init__(self, transparent_ip: str) -> None:
        """
        transparent_ip: the IP assigned to your transparent network inside the container
        """
        self.transparent_ip = transparent_ip
        self.port = 9992
        self.multicast_group_ip = "239.255.42.99"
        self.frequency_in_seconds = 1 
        self.name = "Backend1"

        if len(self.name) > 50:
            self.name = self.name[:40]

        self.stop_event = threading.Event()

    def get_ipv4_interfaces(self):
        ips = []
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == AF_INET:
                    # 127. is the loopback address
                    if not addr.address.startswith("127."):
                        ips.append(addr.address)
        return ips
    
    def send_packets(self):
                
        # HOST_IP = "host.docker.internal"  # Docker Desktop shortcut to host
        # HOST_PORT = 50000                  # must match relay LISTEN_PORT

        # s = socket(AF_INET, SOCK_DGRAM)
        # while True:
        #     message = "CTH" + json.dumps({
        #         "msg_type": "backend_discovery", 
        #         "name": self.name, 
        #         "ip": self.transparent_ip, 
        #         "port": self.port
        #     })
                
        #     if len(message) > 1200:
        #         print("Multicast message is larger than 1200 bytes, will not send!!!")
        #         continue

        #     s.sendto(message.encode("utf-8"), (HOST_IP, HOST_PORT))
        #     print("Sent to host relay")
        #     time.sleep(1)


        while not self.stop_event.is_set():
            try:
                message = "CTH" + json.dumps({
                    "msg_type": "backend_discovery", 
                    "name": self.name, 
                    "ip": self.transparent_ip, 
                    "port": self.port
                })
                
                if len(message) > 1200:
                    print("Multicast message is larger than 1200 bytes, will not send!!!")

                # create UDP socket
                s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
                # set TTL
                s.setsockopt(IPPROTO_IP, IP_MULTICAST_TTL, 1)
                # bind to transparent interface
                s.setsockopt(IPPROTO_IP, IP_MULTICAST_IF, inet_aton(self.transparent_ip))
                # send multicast
                s.sendto(message.encode("utf-8"), (self.multicast_group_ip, self.port))
                s.close()
                print(f"Sent multicast from {self.transparent_ip}")
            except Exception as e:
                print(f"Failed to send: {e}")
            
            time.sleep(self.frequency_in_seconds)
    
    def stop(self):
        self.stop_event.set()
