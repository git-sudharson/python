import socket
import time
serversocket=socket.socket()
host=socket.gethostname()
port=15000
serversocket.bind((host,port))
serversocket.listen(5)
while True:
    clientsocket,addr=serversocket.accept()
    print('got con from',addr)
    currentTime=time.ctime(time.time())+"\r\n"
    clientsocket.send(currentTime.encode('ascii'))
    clientsocket.close()
