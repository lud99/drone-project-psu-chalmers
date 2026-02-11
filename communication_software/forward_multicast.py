import socket
import threading
import psutil

# Configuration
LISTEN_PORT = 50000 # unicast port container sends to
MULTICAST_GROUP = "239.255.42.99"
MULTICAST_PORT = 9992 # original multicast port

# UDP socket to receive from container
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(("0.0.0.0", LISTEN_PORT)) # change to 127.0.0.0 ?

print(f"Relay running: receiving on {LISTEN_PORT}, sending to {MULTICAST_GROUP}:{MULTICAST_PORT}")


def get_ipv4_interfaces():
    ips = []
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                if not addr.address.startswith("127."):
                    ips.append(addr.address)
    return ips

ips = get_ipv4_interfaces()

def relay():
    while True:
        data, addr = recv_sock.recvfrom(4096)
        print(f"Received {len(data)} bytes from {addr}, forwarding to multicast")

        # We must forward on all interfaces to ensure the packet arrives
        for ip in ips:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
                s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
                s.sendto(data, (MULTICAST_GROUP, MULTICAST_PORT))
                s.close()
            # Ignore it, it's okay
            except Exception as _e:
                pass
            
threading.Thread(target=relay, daemon=True).start()

# Keep alive
input("Forwarding multicast packets, press Enter to exit...\n")
