from socket import *
import psutil



def get_ipv4_interfaces():
    ips = []
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == AF_INET:
                if not addr.address.startswith("127."):
                    ips.append(addr.address)
    return ips

print(get_ipv4_interfaces())

MCAST_GRP = "239.255.42.99"
PORT = 9991

s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
s.setsockopt(IPPROTO_IP, IP_MULTICAST_TTL, 1)

for ip in get_ipv4_interfaces():
    try:
        s.setsockopt(IPPROTO_IP, IP_MULTICAST_IF, inet_aton(ip))
        s.sendto(b"hello", ("239.255.42.99", 9991))
    except:
        pass