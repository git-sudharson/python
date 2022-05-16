import socket
s=socket.socket()
host=socket.gethostname()
port=15000
s.connect((host,port))
print(s.recv(1024))
tm=s.recv(1024)
s.close()
print("The time got from the server is %s"%tm.decode('ascii'))
