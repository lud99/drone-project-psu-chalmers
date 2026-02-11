import socket
import struct
import psutil

MCAST_GRP = "239.255.42.99"
PORT = 9992

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('', PORT))

def get_ipv4_interfaces():
    ips = []
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                if not addr.address.startswith("127."):
                    ips.append(addr.address)
    return ips

# Join multicast group on all interfaces
for ip in get_ipv4_interfaces():
    try:
        mreq = struct.pack(
            "=4s4s",
            socket.inet_aton(MCAST_GRP),
            socket.inet_aton(ip)
        )
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        print("lsitening on " + ip)
    except Exception as _e:
        pass


print("listening (multicast)...")
while True:
    data, addr = s.recvfrom(1024)
    print(data, addr)
