"""
串口通讯模块
保留 MTI 的 19 字节数据帧格式
"""

import serial
import serial.tools.list_ports


class SerialCommunicator:
    def __init__(self):
        self.ser = None

    def list_ports(self):
        """列出可用串口"""
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("无可用串口设备")
            return []
        print("可用串口:")
        for p in ports:
            print(f"  {p.device} - {p.description}")
        return ports

    def open(self, port="/dev/ttyUSB0", baudrate=115200):
        """打开串口

        参数:
            port: 串口名称
            baudrate: 波特率
        """
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=0.1
            )
            print(f"串口已打开: {port} @ {baudrate}")
            return True
        except Exception as e:
            print(f"打开串口失败: {e}")
            return False

    def send(self, msg):
        """发送数据帧

        参数:
            msg: 数据列表，每项为 0-255 的整数
        """
        if self.ser is None or not self.ser.is_open:
            return
        for data in msg:
            self.ser.write(bytes([data]))

    def close(self):
        """关闭串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None
            print("串口已关闭")
