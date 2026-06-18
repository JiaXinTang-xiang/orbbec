"""
PID 控制器模块
用于目标追踪时的运动控制
"""


class PIDController:
    def __init__(self, kp=0.5, ki=0.0, kd=0.2, deadband=0.0, max_output=255, min_output=0):
        """初始化 PID 控制器

        参数:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
            deadband: 死区范围，误差在此范围内不调整
            max_output: 输出最大值
            min_output: 输出最小值
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.deadband = deadband
        self.max_output = max_output
        self.min_output = min_output

        self.error_sum = 0
        self.last_error = 0

    def update(self, current, target=0.0):
        """计算 PID 输出

        参数:
            current: 当前值
            target: 目标值

        返回:
            PID 输出值
        """
        error = target - current

        # 死区内不调整
        if abs(error) <= self.deadband:
            self.error_sum = 0
            return 0

        # P 项
        p_term = self.kp * error

        # I 项
        self.error_sum += error
        i_term = self.ki * self.error_sum

        # D 项
        d_term = self.kd * (error - self.last_error)
        self.last_error = error

        # 总输出
        output = p_term + i_term + d_term

        # 限幅
        output = max(self.min_output, min(self.max_output, output))

        return output

    def reset(self):
        """重置控制器状态"""
        self.error_sum = 0
        self.last_error = 0
