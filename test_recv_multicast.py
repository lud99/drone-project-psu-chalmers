from socket import *
import struct

MCAST_GRP = "239.255.42.99"
PORT = 9991

s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
s.bind(('', PORT))

# Join multicast group on all interfaces
mreq = struct.pack("4sl", inet_aton(MCAST_GRP), INADDR_ANY)
s.setsockopt(IPPROTO_IP, IP_ADD_MEMBERSHIP, mreq)

print("listening (multicast)...")
while True:
    data, addr = s.recvfrom(1024)
    print(data, addr)
