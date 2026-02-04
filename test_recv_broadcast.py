from socket import *

s = socket(AF_INET, SOCK_DGRAM)
s.bind(('', 999))

print(s.recvfrom(1024))