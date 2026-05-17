import asyncio
import signal
import struct
import threading
from functools import reduce

from dpg_system.node import Node
from dpg_system.conversion_utils import *

from bleak import BleakClient
from bleak import _logger as logger
from bleak import discover


def register_movesense_nodes():
    Node.app.register_node('movesense', MoveSenseNode.factory)


WRITE_CHARACTERISTIC_UUID = (
    "34800001-7185-4d5d-b431-630e7050e8f0"
)

NOTIFY_CHARACTERISTIC_UUID = (
    "34800002-7185-4d5d-b431-630e7050e8f0"
)

# Per-sample payload is [type byte, timestamp (uint32 at offset 2),
# ax/ay/az (float32 at offsets 6/10/14)]. Bytes 0..17 inclusive.
_ACC_PAYLOAD_MIN_LEN = 18


class DataView:
    def __init__(self, array, bytes_per_element=1):
        """
        bytes_per_element is the size of each element in bytes.
        By default we assume the array is one byte per element.
        """
        self.array = array
        self.bytes_per_element = bytes_per_element

    def __get_binary(self, start_index, byte_count, signed=False):
        integers = [self.array[start_index + x] for x in range(byte_count)]
        as_bytes = [integer.to_bytes(
            self.bytes_per_element, byteorder='little', signed=signed) for integer in integers]
        return reduce(lambda a, b: a + b, as_bytes)

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
    @staticmethod
    def factory(name, data, args=None):
        node = MoveSenseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        args = args or []
        self.end_of_serial = '000431'
        if len(args) > 0:
            self.end_of_serial = any_to_string(args[0])
        self.output_accelerometer = self.add_output('accelerometer')
        self.output_gyroscope = self.add_output('gyroscope')
        self.output_magnetometer = self.add_output('magnetometer')

        self.queue = None
        self.client_task = None
        self.consumer_task = None
        # NB: BLE discovery and the client task are async (see init_queue /
        # run_ble_client below). They are not wired up to the node yet —
        # the previous synchronous discover() call here couldn't work
        # because discover() is a coroutine function; the result was
        # discarded immediately even when it didn't outright raise.


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
            logger.info("received: " + str(data))


def _install_signal_handler(handler):
    """signal.signal only works on the main thread; in a worker thread it
    raises ValueError. Swallow that so the BLE client survives."""
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError) as e:
        logger.info("could not install signal handlers: %s", e)


async def run_ble_client(end_of_serial: str, queue: asyncio.Queue):
    address = None
    try:
        devices = await discover()
    except Exception as e:
        logger.info("BLE discover failed: %s", e)
        await queue.put(None)
        return

    for d in devices:
        print("device:", d)
        if d.name and d.name.endswith(end_of_serial):
            print("device found")
            address = d.address
            break

    if address is None:
        await queue.put(None)
        print("Sensor  ******" + end_of_serial, "not found!")
        return

    # This event is set if device disconnects or ctrl+c is pressed
    disconnected_event = asyncio.Event()

    def raise_graceful_exit(*args):
        disconnected_event.set()

    def disconnect_callback(client):
        logger.info("Disconnected callback called!")
        disconnected_event.set()

    async def notification_handler(sender, data):
        """Simple notification handler which prints the data received."""
        if data is None or len(data) < _ACC_PAYLOAD_MIN_LEN:
            return
        try:
            d = DataView(data)
            msg = "Data: ts: {}, ax: {}, ay: {}, az: {}".format(
                d.get_uint_32(2),
                d.get_float_32(6),
                d.get_float_32(10),
                d.get_float_32(14),
            )
        except Exception as e:
            logger.info("notification parse failed: %s", e)
            return
        try:
            await queue.put(msg)
        except Exception as e:
            logger.info("queue put failed: %s", e)

    try:
        async with BleakClient(address, disconnected_callback=disconnect_callback) as client:
            _install_signal_handler(raise_graceful_exit)

            # Start notifications and subscribe to acceleration @ 13Hz
            logger.info("Enabling notifications")
            await client.start_notify(NOTIFY_CHARACTERISTIC_UUID, notification_handler)
            logger.info("Subscribing datastream")
            await client.write_gatt_char(
                WRITE_CHARACTERISTIC_UUID,
                bytearray([1, 99]) + bytearray("/Meas/Acc/13", "utf-8"),
                response=True,
            )

            # Run until disconnect event is set
            await disconnected_event.wait()
            logger.info(
                "Disconnect set by ctrl+c or real disconnect event. Check Status:")

            # Check the connection status to infer if the device disconnected or ctrl+c was pressed
            try:
                status = client.is_connected
            except Exception as e:
                logger.info("is_connected check failed: %s", e)
                status = False
            logger.info("Connected: {}".format(status))

            # If status is connected, unsubscribe and stop notifications
            if status:
                try:
                    logger.info("Unsubscribe")
                    await client.write_gatt_char(WRITE_CHARACTERISTIC_UUID, bytearray([2, 99]), response=True)
                    logger.info("Stop notifications")
                    await client.stop_notify(NOTIFY_CHARACTERISTIC_UUID)
                except Exception as e:
                    logger.info("teardown write failed: %s", e)

            await asyncio.sleep(0.1)
    except Exception as e:
        logger.info("BLE client session failed: %s", e)
    finally:
        # Signal consumer to exit
        try:
            await queue.put(None)
        except Exception:
            pass
