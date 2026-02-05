from socket import *
import struct
import psutil

MCAST_GRP = "239.255.42.99"
PORT = 9992

s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
s.bind(('', PORT))

def get_ipv4_interfaces():
    ips = []
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == AF_INET:
                if not addr.address.startswith("127."):
                    ips.append(addr.address)
    return ips

# Join multicast group on all interfaces
for ip in get_ipv4_interfaces():
    try:
        mreq = struct.pack(
            "=4s4s",
            inet_aton(MCAST_GRP),
            inet_aton(ip)
        )
        s.setsockopt(IPPROTO_IP, IP_ADD_MEMBERSHIP, mreq)
        print("lsitening on " + ip)
    except:
        pass


print("listening (multicast)...")
while True:
    data, addr = s.recvfrom(1024)
    print(data, addr)
