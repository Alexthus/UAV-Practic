import numpy as np
import constants as cs
from constants import FlightMode, States 
from pid import PID

class QuadCopterController:
    def __init__(self):
        
        self.target_x = 0
        self.target_y = 0
        self.target_z = 10
        self.target_yaw = 0
        
        self.position_controller_x = PID()
        self.position_controller_y = PID()
        self.position_controller_z = PID()

        self.velocity_controller_x = PID()
        self.velocity_controller_y = PID()
        self.velocity_controller_z = PID()

        self.roll_controller = PID()
        self.pitch_controller = PID()
        self.yaw_controller = PID()

        self.roll_rate_controller = PID()
        self.pitch_rate_controller = PID()
        self.yaw_rate_controller = PID()

        # Внешние контуры — позиция (мягкие)
        self.position_controller_x.set_pid_gains(0.5, 0.0, 0.1)
        self.position_controller_y.set_pid_gains(0.5, 0.0, 0.1)
        self.position_controller_z.set_pid_gains(0.5, 0.0, 0.2)

        # Внутренние — скорость (энергичные)
        self.velocity_controller_x.set_pid_gains(2.0, 0.1, 0.2)
        self.velocity_controller_y.set_pid_gains(2.0, 0.1, 0.2)
        self.velocity_controller_z.set_pid_gains(1.5, 0.1, 0.1)

        # Углы
        self.roll_controller.set_pid_gains(3.0, 0.1, 0.2)
        self.pitch_controller.set_pid_gains(3.0, 0.1, 0.2)

        # Угловая скорость — самые быстрые
        self.roll_rate_controller.set_pid_gains(20.0, 1.0, 0.5)
        self.pitch_rate_controller.set_pid_gains(20.0, 1.0, 0.5)
        self.yaw_rate_controller.set_pid_gains(15.0, 0.5, 0.3)

        # Пример простого маршрута
        self._mission = [[0, 0, 10, 0], [5, 0, 10, 0]]
        self._current_mission_index = 0


    def set_target_position(self, x, y, z, yaw):
        self.target_x = x
        self.target_y = y
        self.target_z = z
        self.target_yaw = yaw
            # Сброс интегралов для всех основных контуров
        self.velocity_controller_z._integral = 0.0
        self.velocity_controller_x._integral = 0.0
        self.velocity_controller_y._integral = 0.0


    def update(self, state_vector, dt) -> np.ndarray:

        #TODO Обновление целевой точки маршрута
        # Получаем текущую позицию из state_vector
        current_x = state_vector[States.X]
        current_y = state_vector[States.Y]
        current_z = state_vector[States.Z]

        # Проверяем достижение текущей цели
        pos_tolerance = 0.3  # метров
        if (abs(current_x - self.target_x) < pos_tolerance and
            abs(current_y - self.target_y) < pos_tolerance and
            abs(current_z - self.target_z) < pos_tolerance):

            # Переход к следующей точке маршрута
            if self._current_mission_index < len(self._mission) - 1:
                self._current_mission_index += 1
                wp = self._mission[self._current_mission_index]
                self.set_target_position(wp[0], wp[1], wp[2], wp[3])
                # Иначе — остаёмся на последней точке

        #TODO Расчет целевой скорости ЛА
        # Целевые скорости (выходы PID-регуляторов положения)
        v_des_x = self.position_controller_x.update(current_x, self.target_x, dt)
        v_des_y = self.position_controller_y.update(current_y, self.target_y, dt)
        v_des_z = self.position_controller_z.update(current_z, self.target_z, dt)


        # ограничения скорости
        v_des_x = np.clip(v_des_x, -1.5, 1.5)
        v_des_y = np.clip(v_des_y, -1.5, 1.5)
        v_des_z = np.clip(v_des_z, -1.0, 2.0)  # вверх быстрее, чем вниз

        #TODO Расчет тяги и углов крена и тангажа, перепроектирование углов в связную СК
        current_vx = state_vector[States.VX]
        current_vy = state_vector[States.VY]
        current_vz = state_vector[States.VZ]

        current_yaw = float(state_vector[States.YAW])
        R = self._rotation2d(-current_yaw)
        v_des_body = R @ np.array([float(v_des_x), float(v_des_y)])
        v_des_x_body, v_des_y_body = v_des_body


        current_v_body = R @ np.array([current_vx, current_vy])
        
        
        target_pitch = self.velocity_controller_x.update(current_v_body[0], v_des_x_body, dt)
        target_roll = self.velocity_controller_y.update(current_v_body[1], v_des_y_body, dt)
        # Желаемые тяги и углов крена и тангажа на выходе ПИД скорости
        #target_pitch = self.velocity_controller_x.update(current_vx, v_des_x_body, dt)
        #target_roll = self.velocity_controller_y.update(current_vy, v_des_y_body, dt)
        #cmd_trust = self.velocity_controller_z.update(current_vz, v_des_z, dt)

        print(f"target_pitch: {np.degrees(target_pitch).item():.1f}°, target_roll: {np.degrees(target_roll).item():.1f}°")
        
        # Ограничиваем желаемые углы разумными значениями (в радианах)
        max_angle = np.radians(5)  # 5 градусов — максимум для устойчивого полёта
        target_pitch = np.clip(target_pitch, -max_angle, max_angle)
        target_roll = np.clip(target_roll, -max_angle, max_angle)
        print(f"target_pitch_afterclip: {np.degrees(target_pitch).item():.1f}°, target_roll_afterclip: {np.degrees(target_roll).item():.1f}°")

        # Желаемая суммарная тяга (в Ньютонах) для расчёта висения
        hover_thrust_total = cs.quadcopter_mass * cs.GRAVITY
        thrust_adjust_N = self.velocity_controller_z.update(current_vz, v_des_z, dt)  # в Ньютонах!
        thrust_adjust_N = np.clip(thrust_adjust_N, -1.0, 1.0)
        desired_thrust = hover_thrust_total + thrust_adjust_N
        desired_thrust = np.clip(desired_thrust, 0.1, 4 * cs.trust_coef * (cs.max_rotors_rpm ** 2))
        # Базовая RPM для общей тяги
        base_rpm = np.sqrt(desired_thrust / (4 * cs.trust_coef))

        print(f"desired_thrust: {desired_thrust.item():.2f} N, hover: {cs.quadcopter_mass * cs.GRAVITY:.2f} N")
        print(f"base_rpm: {base_rpm.item():.2f}")

        # Пример для контура управления угловым положением и угловой скоростью.
        target_roll_rate = self.roll_controller.update(state_vector[States.ROLL], target_roll, dt)
        target_pitch_rate = self.pitch_controller.update(state_vector[States.PITCH], target_pitch, dt)
        target_yaw_rate = self.yaw_controller.update(state_vector[States.YAW], self.target_yaw, dt)

        cmd_roll = self.roll_rate_controller.update(state_vector[States.ROLL_RATE], target_roll_rate, dt)
        cmd_pitch = self.pitch_rate_controller.update(state_vector[States.PITCH_RATE], target_pitch_rate, dt)
        cmd_yaw = self.yaw_rate_controller.update(state_vector[States.YAW_RATE], target_yaw_rate, dt)

        u = self._mixer(base_rpm, cmd_roll, cmd_pitch, cmd_yaw)
        return u

    def _mixer(self, cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw) -> np.ndarray:

        # Команды: thrust (общая тяга), roll (крен), pitch (тангаж), yaw (рыскание)
        # Смешивание команд
        u0 = cmd_thrust - cmd_pitch - cmd_yaw   # front
        u1 = cmd_thrust - cmd_roll  + cmd_yaw   # right
        u2 = cmd_thrust + cmd_pitch - cmd_yaw   # back
        u3 = cmd_thrust + cmd_roll  + cmd_yaw   # left

        max_cmd = 3000.0
        return np.clip([u0, u1, u2, u3], 0, max_cmd)

    def _rotation2d(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta),  np.cos(theta)]]) 
