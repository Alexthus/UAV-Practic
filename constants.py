from enum import IntEnum, unique
import numpy as np

@unique
class FlightMode(IntEnum):
	POSITION_HOLD = 1
	VELOCITY = 2
	ANGLE = 3
	ACRO = 4

@unique
class States(IntEnum):
	X = 0
	Y = 1
	Z = 2
	ROLL = 3
	PITCH = 4
	YAW = 5
	VX = 6
	VY = 7
	VZ = 8
	ROLL_RATE = 9
	PITCH_RATE = 10
	YAW_RATE = 11

arm_length = 0.17
trust_coef = 3.9865e-06 # 3.9865e-08
drag_coef = 7.5e-9

max_rotors_rpm = 3500
min_rotors_rpm = 100
trust_scale = 1000

quadcopter_mass = 1 # 0.0630
quadcopter_inertia = np.array([[5.82857e-04, 0.0, 0.0], 
              				   [0.0, 7.16914e-04, 0.0], 
              				   [0.0, 0.0, 0.01]])

GRAVITY = 9.81