import socket
import struct
import queue
import threading
from typing import Any
import time

class ConcurrentQueue:
    def __init__(self):
        self.push_id = 0
        self.pop_id = 0
        self.queue = queue.Queue()

    def push(self, obj: Any) -> None:
        current_time_millis = int(round(time.time() * 1000))
        print("pushed, ", self.push_id, " timestamp is, ", current_time_millis)
        self.push_id += 1
        self.queue.put(obj)

    def pop(self) -> Any:
        res = self.queue.get()
        current_time_millis = int(round(time.time() * 1000))
        print("pop, ", self.pop_id, " timestamp is, ", current_time_millis)
        self.pop_id += 1
        return res

    def size(self) -> int:
        return self.queue.qsize()


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


class PacketDeserializerWorker:
    def __init__(self, client_socket: socket.socket, pipe: ConcurrentQueue):
        self.client_socket = client_socket
        self.pipe = pipe

    def run(self):
        while True:
            header_data = recvall(self.client_socket, 10)
            if not header_data:
                break

            type, size, chksum = struct.unpack('!IIH', header_data)

            if chksum != 0xABCD:
                continue

            buffer = recvall(self.client_socket, size)
            chksum2, = struct.unpack('!H', recvall(self.client_socket, 2))

            if chksum2 != 0xDCBA:
                continue

            packet = {"type": type, "size": size, "payload": buffer}
            self.pipe.push(packet)


class PacketDeserializer:
    def __init__(self, port: int):
        self.port = port
        self.listener = None
        self.listenerThread = None
        self.pipe = ConcurrentQueue()

    def start(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(('0.0.0.0', self.port))
        self.listener.listen(5)

        self.listenerThread = threading.Thread(target=self.run, daemon=True)
        self.listenerThread.start()
        return 0

    def run(self):
        while True:
            client_socket, _ = self.listener.accept()
            worker = PacketDeserializerWorker(client_socket, self.pipe)
            threading.Thread(target=worker.run, daemon=True).start()

    def read(self) -> dict:
        return self.pipe.pop()
