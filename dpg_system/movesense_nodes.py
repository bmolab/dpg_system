import dearpygui.dearpygui as dpg
import math
import numpy as np
import random
import time
from dpg_system.node import Node
import threading
from dpg_system.conversion_utils import *
import json
from fuzzywuzzy import fuzz
import asyncio
import signal
import struct
from bleak import BleakClient
from bleak import _logger as logger
from bleak import discover
from functools import reduce

def register_movesense_nodes():
    Node.app.register_node('movesense', MoveSenseNode.factory)


WRITE_CHARACTERISTIC_UUID = (
    "34800001-7185-4d5d-b431-630e7050e8f0"
)

NOTIFY_CHARACTERISTIC_UUID = (
    "34800002-7185-4d5d-b431-630e7050e8f0"
)


class DataView:
    def __init__(self, array, bytes_per_element=1):
        """
        bytes_per_element is the size of each element in bytes.
        By default we are assume the array is one byte per element.
        """
        self.array = array
        self.bytes_per_element = 1

    def __get_binary(self, start_index, byte_count, signed=False):
        integers = [self.array[start_index + x] for x in range(byte_count)]
        bytes = [integer.to_bytes(
            self.bytes_per_element, byteorder='little', signed=signed) for integer in integers]
        return reduce(lambda a, b: a + b, bytes)

    def get_uint_16(self, start_index):
        bytes_to_read = 2
        return int.from_bytes(self.__get_binary(start_index, bytes_to_read), byteorder='little')

    def get_uint_8(self, start_index):
        bytes_to_read = 1
        return int.from_bytes(self.__get_binary(start_index, bytes_to_read), byteorder='little')

    def get_uint_32(self, start_index):
        bytes_to_read = 4
        binary = self.__get_binary(start_index, bytes_to_read)
        return struct.unpack('<I', binary)[0]  # <f for little endian

    def get_float_32(self, start_index):
        bytes_to_read = 4
        binary = self.__get_binary(start_index, bytes_to_read)
        return struct.unpack('<f', binary)[0]  # <f for little endian


class MoveSenseNode(Node):
    # comment_theme = None
    # inited = False

    @staticmethod
    def factory(name, data, args=None):
        node = MoveSenseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.end_of_serial = '000431'
        if len(args) > 0:
            self.end_of_serial = any_to_string(args[0])
        self.output_accelerometer = self.add_output('accelerometer')
        self.output_gyroscope = self.add_output('gyroscope')
        self.output_magnetometer = self.add_output('magnetometer')

        self.queue = None
        self.client_task = None
        self.consumer_task = None
        devices = discover()
        found = False
        address = None
        for d in devices:
            print("device:", d)
            if d.name and d.name.endswith(self.end_of_serial):
                print("device found")
                address = d.address
                found = True
                break


async def init_queue(node):
    node.queue = asyncio.Queue()
    node.client_task = run_ble_client(node.end_of_serial, node.queue)
    node.consumer_task = run_queue_consumer(node.queue)
    await asyncio.gather(node.client_task, node.consumer_task)


async def run_queue_consumer(queue: asyncio.Queue):
    while True:
        data = await queue.get()
        if data is None:
            logger.info(
                "Got message from client about disconnection. Exiting consumer loop..."
            )
            break
        else:
            logger.info("received: " + data)


async def run_ble_client(end_of_serial: str, queue: asyncio.Queue):
    # Check the device is available
    devices = await discover()
    found = False
    address = None
    for d in devices:
        print("device:", d)
        if d.name and d.name.endswith(end_of_serial):
            print("device found")
            address = d.address
            found = True
            break

    # This event is set if device disconnects or ctrl+c is pressed
    disconnected_event = asyncio.Event()

    def raise_graceful_exit(*args):
        disconnected_event.set()

    def disconnect_callback(client):
        logger.info("Disconnected callback called!")
        disconnected_event.set()

    async def notification_handler(sender, data):
        """Simple notification handler which prints the data received."""
        d = DataView(data)
        # Dig data from the binary
        msg = "Data: ts: {}, ax: {}, ay: {}, az: {}".format(d.get_uint_32(2), d.get_float_32(6), d.get_float_32(10), d.get_float_32(14))
        # queue message for later consumption
        await queue.put(msg)

    if found:
        async with BleakClient(address, disconnected_callback=disconnect_callback) as client:

            loop = asyncio.get_event_loop()
            # Add signal handler for ctrl+c
            signal.signal(signal.SIGINT, raise_graceful_exit)
            signal.signal(signal.SIGTERM, raise_graceful_exit)

            # Start notifications and subscribe to acceleration @ 13Hz
            logger.info("Enabling notifications")
            await client.start_notify(NOTIFY_CHARACTERISTIC_UUID, notification_handler)
            logger.info("Subscribing datastream")
            await client.write_gatt_char(WRITE_CHARACTERISTIC_UUID, bytearray([1, 99]) + bytearray("/Meas/Acc/13", "utf-8"), response=True)

            # Run until disconnect event is set
            await disconnected_event.wait()
            logger.info(
                "Disconnect set by ctrl+c or real disconnect event. Check Status:")

            # Check the conection status to infer if the device disconnected or crtl+c was pressed
            status = client.is_connected
            logger.info("Connected: {}".format(status))

            # If status is connected, unsubscribe and stop notifications
            if status:
                logger.info("Unsubscribe")
                await client.write_gatt_char(WRITE_CHARACTERISTIC_UUID, bytearray([2, 99]), response=True)
                logger.info("Stop notifications")
                await client.stop_notify(NOTIFY_CHARACTERISTIC_UUID)

            # Signal consumer to exit
            await queue.put(None)

            await asyncio.sleep(0.1)

    else:
        # Signal consumer to exit
        await queue.put(None)
        print("Sensor  ******" + end_of_serial, "not found!")




