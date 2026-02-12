import socket
import psutil
import json


def get_ipv4_interfaces():
    ips = []
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                # if not addr.address.startswith("127."):
                ips.append(addr.address)
    return ips

print(get_ipv4_interfaces())

MCAST_GRP = "239.255.42.99"
PORT = 9992

for ip in get_ipv4_interfaces():
    try:
        message = "CTH" + json.dumps({ "msg_type": "backend_discovery", 
                                          "name": "backend1", 
                                          "ip": ip, 
                                          "port": int(9992)})
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
        s.sendto(message.encode("utf-8"), ("239.255.42.99", 9992))
        s.close()
        
        print("try send " + ip)
    except Exception as _e:
        print(ip + " failed")