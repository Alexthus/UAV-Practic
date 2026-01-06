import numpy as np

class QuadCopterModel:
	def __init__(self, inertia, mass: float, trust_coef: float, drag_coef: float, arm_length: float):
		# Зададим константные параметры модели 
		self._inertia = inertia
		self._inertia_inv = np.linalg.inv(inertia)
		self._mass = mass
		self._trust_coef = trust_coef
		self._drag_coef = drag_coef
		self._arm_length = arm_length
		# Запишем вектор силы тяжести
		self._g = np.array([[0.0], [0.0], [-9.81]])
		self._motor_trust = np.array([[0.0], [0.0], [0.0]])
		self._motor_moments = np.array([[0.0], [0.0], [0.0]])

		# Зададим вектор состояния, запишем начальные значения, в данном случае везде нули.
		self._state_vector = np.array([[0.0],  # pose X
		 							  [0.0],  # pose Y
									  [0.0],  # pose Z
									  [0.0],  # roll 
									  [0.0],  # pitch 
									  [0.0],  # yaw 
									  [0.0],  # velocity X
									  [0.0],  # velocity Y
									  [0.0],  # velocity Z
									  [0.0],  # roll rate 
									  [0.0],  # pitch rate   
									  [0.0]]) # yaw rate

	@property
	def state_vector(self):
		return self._state_vector

	def update_state(self, u, dt: float) -> None:
		# Получим линейное и угловое ускорение.
		lin_acc, ang_acc = self._func_right(u)
		# Проинтегрируем полученные приращения и обновим вектор состояния
		self._integrate(lin_acc, ang_acc, dt)

	def _integrate(self, linear_acceleration, angular_acceleration, dt: float):
		# Проинтегрируем линейное ускорение и обновим скорость
		self.state_vector[6:9] += linear_acceleration * dt
		# Проинтегрируем скорость и обновим положение
		self.state_vector[0:3] += self.state_vector[6:9] * dt
		# Проинтегрируем угловое ускорение и обновим угловую скорость
		self.state_vector[9:12] += angular_acceleration * dt
		# Проинтегрируем угловую скорость и обновим угловое положение ЛА
		self.state_vector[3:6] += self.state_vector[9:12] * dt

	def _func_right(self, u):
		
		u = np.ravel(u)

		# Рассчитаем линейное ускорение по управляющим сигналам u = [u1, u2, u3, u4]
        # Извлекаем углы из вектора состояния
		roll = self.state_vector[3, 0]
		pitch = self.state_vector[4, 0]
		yaw = self.state_vector[5, 0]

        # Матрица поворота из связанной СК в инерциальную
		R = self._rotation_matrix_3d(pitch, roll, yaw)

        # Тяга двигателей: u_i * trust_coef
		thrusts = (u ** 2) * self._trust_coef  # shape: (4,)
		total_thrust = np.sum(thrusts)  # скаляр

        # Линейное ускорение: a = R * [0, 0, ΣP_i] / m + g
		F_body = np.array([[0.0], [0.0], [total_thrust]]) # shape (3,1)
		F_inertial = R @ F_body # shape (3,1)
		linear_acceleration = F_inertial / self._mass + self._g # shape (3,1) linear acceleration = [a1, a2, a3]

		# Рассчитываем моменты создаваемые двигателями в связной СК
        # Предполагаем X-конфигурацию рамы:
        #   Двигатель 0: +X (по часовой стрелке)
        #   Двигатель 1: +Y (против часовой)
        #   Двигатель 2: -X (по часовой)
        #   Двигатель 3: -Y (против часовой)
	
		M_x = self._arm_length * self._trust_coef * (u[0] ** 2 - u[2] ** 2)           # Roll moment
		M_y = self._arm_length * self._trust_coef * (u[3] ** 2 - u[1] ** 2)           # Pitch moment
		M_z = self._drag_coef * (u[3] ** 2 + u[1] ** 2 - u[0] ** 2 - u[2] ** 2)  # Yaw moment
		
		moments = np.array([[M_x], [M_y], [M_z]]) # shape (3,1)

        # Угловая скорость в связанной СК
		omega = self.state_vector[9:12]  # shape: (3,1)

        # Гироскопический член: omega × (I * omega)
		I_omega = self._inertia @ omega
		gyro_term = np.cross(omega.flatten(), I_omega.flatten()).reshape((3,1))

        # Угловое ускорение: ω_dot = I⁻¹ * (M - gyro_term)
		angular_acceleration = self._inertia_inv @ (moments - gyro_term) # shape (3,1)
		
		return linear_acceleration, angular_acceleration

	def _rotation_matrix_3d(self, pitch, roll, yaw):
		return np.array([[np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.cos(roll), -np.sin(roll)],
						  [np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(pitch),
						  	np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(pitch),
							np.sin(pitch)*np.cos(roll)],
						  [np.cos(yaw)*np.sin(roll)*np.cos(pitch) + np.sin(yaw)*np.sin(pitch),
							np.sin(yaw)*np.cos(pitch)*np.sin(roll) - np.cos(yaw)*np.sin(pitch),
							np.cos(pitch)*np.cos(roll)]])

		
