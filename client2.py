#client2.py
# lets make the client code
import socket,cv2, pickle,struct
import numpy as np

# create socket

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.4.1' # paste your server ip address here
port = 5000
print("Connecting to socket on {}:{}...".format(host_ip, port))
client_socket.connect((host_ip,port)) # a tuple
print("Connection Successful")

imgdata = None
data_buffer = b''


while True:
    '''authenticated connection to IP cam'''
    data_buffer+=(client_socket.recv(512))
    start_idx = data_buffer.find(b'\xff\xd8')
    end_idx = data_buffer.find(b'\xff\xd9')
    # print("found begin: ", a, " found end: ", b)
    # i = np.zeros()

    # At startup we might get an end before we get the first start, if
    # that is the case then throw away the data before start
    if end_idx > -1 and end_idx < start_idx:
        data_buffer = data_buffer[start_idx:]

    # We have a start and an end of the image in the buffer now
    if start_idx > -1 and end_idx > -1 and end_idx > start_idx:
        # Pick out the image to render ...
        imgdata = data_buffer[start_idx:end_idx + 2]
        # .. and remove it from the buffer
        data_buffer = data_buffer[end_idx + 2 :]
        print(imgdata)
        i = cv2.imdecode(np.fromstring(imgdata, dtype=np.uint8),cv2.IMREAD_GRAYSCALE)

        cv2.imshow('authenticated cam',i)
        if cv2.waitKey(1) ==27:
            exit(0)

# Basic Socket Read, not compatitble with AI-Deck

# data = b""
# payload_size = struct.calcsize("Q")

# 	while len(data) < payload_size:
# 		packet = client_socket.recv(512) # buffer size from crazyflie
# 		if not packet: break
# 		data+=packet
# 	packed_msg_size = data[:payload_size]
# 	data = data[payload_size:]
# 	msg_size = struct.unpack("Q",packed_msg_size)[0]

# Basic Socket Read, not compatitble with AI-Deck
    # msg_size = end_idx - start_idx
    # while len(data) < msg_size:
    #     data += client_socket.recv(512)
    # frame_data = data[:msg_size]
    # data  = data[msg_size:]
    # frame = pickle.loads(imgdata)
    # cv2.imshow("RECEIVING VIDEO",frame)
    # key = cv2.waitKey(1) & 0xFF
    # if key  == ord('q'):
    #     break


client_socket.close()