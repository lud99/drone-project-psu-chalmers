from socket import *
import json
import psutil
import threading
import time

class MulticastSender:
    def __init__(self) -> None:
        self.list_of_ips = self.get_ipv4_interfaces()
        self.port = 9992
        self.multicast_group_ip = "239.255.42.99"
        self.frequency_in_seconds = 5 
        self.name = "Backend1" # TODO implement properly

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
        while not self.stop_event.is_set():
            for ip in self.list_of_ips:
                try:
                    message = "CTH" + json.dumps({ "msg_type": "backend_discovery", 
                                          "name": self.name, 
                                          "ip": ip, 
                                          "port": self.port})
                    s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
                    s.setsockopt(IPPROTO_IP, IP_MULTICAST_TTL, 1)
                    s.setsockopt(IPPROTO_IP, IP_MULTICAST_IF, inet_aton(ip))
                    s.sendto(message.encode("utf-8"), (self.multicast_group_ip, self.port))
                    s.close()
                    
                    print("try send " + ip)
                except Exception as e:
                    print(f"{ip} failed: {e}")
            
            time.sleep(self.frequency_in_seconds)

    def stop(self):
        self.stop_event.set()
