#!/usr/bin/env python3

import socket
import logging
import numpy as np
from io import BytesIO


class NumpySocket(socket.socket):
    def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
        super().__init__(family, type, proto, fileno)
        self.remaining_buffer = None

    def sendall(self, frame):
        if not isinstance(frame, np.ndarray):
            raise TypeError("input frame is not a valid numpy array")  # should this just call super intead?

        out = self.__pack_frame(frame)
        super().sendall(out)
        logging.debug("frame sent")

    def recv(self, bufsize=1024):
        length = None
        frameBuffer = None
        if self.remaining_buffer is not None:
            frameBuffer = self.remaining_buffer
            self.remaining_buffer = None
        else:
            frameBuffer = bytearray()

        while True:
            data = np.array([])
            try:
                data = super().recv(bufsize)
            except Exception as e:
                if length is None:
                    raise e

            if len(data) == 0:
                return np.array([])
            frameBuffer += data
            # print(len(frameBuffer))

            if len(frameBuffer) == length:
                break

            loop = True
            while loop:
                if length is None:
                    if b':' not in frameBuffer:
                        break
                    # remove the length bytes from the front of frameBuffer
                    # leave any remaining bytes in the frameBuffer!
                    length_str, ignored, frameBuffer = frameBuffer.partition(b':')
                    if len(length_str) > 16:
                        print('recv length_str longer than 16 bytes')
                    else:
                        try:
                            length = int(length_str)
                        except Exception as e:
                            print('recv length_str does not convert to int')

                # print(len(frameBuffer), end='')
                if len(frameBuffer) < length:
                    break
                # split off the full message from the remaining bytes
                # leave any remaining bytes in the frameBuffer!
                # print('length or more', len(frameBuffer), length)

                if len(frameBuffer) > length:
                    self.remaining_buffer = frameBuffer[length:]
                    # print('remainder', self.remaining_buffer)
                frameBuffer = frameBuffer[:length]
                length = None
                loop = False

            if not loop:
                break

        frame = np.load(BytesIO(frameBuffer), allow_pickle=True)['frame']
        logging.debug("frame received")
        return frame

    def accept(self):
        fd, addr = super()._accept()
        sock = NumpySocket(super().family, super().type, super().proto, fileno=fd)
        
        if socket.getdefaulttimeout() is None and super().gettimeout():
            sock.setblocking(True)
        return sock, addr
    

    @staticmethod
    def __pack_frame(frame):
        f = BytesIO()
        np.savez(f, frame=frame)
        
        packet_size = len(f.getvalue())
        header = '{0}:'.format(packet_size)
        header = bytes(header.encode())  # prepend length of array

        out = bytearray()
        out += header

        f.seek(0)
        out += f.read()
        return out
