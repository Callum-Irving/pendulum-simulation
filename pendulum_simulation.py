from numba import jit
import matplotlib.pyplot as plt
import numpy as np

time = 0
mass = 1
G = 0.1
length = 40
initial_angle = np.pi / 4


@jit(nopython=True)
def simulate(mass, G, length, angle, damping, num_iter):
    positions = np.zeros(num_iter, dtype=np.float32)
    vel = 0

    for i in range(num_iter):
        positions[i] = angle
        force = -mass * G * np.sin(angle) / length
        vel += force
        vel *= damping
        angle += vel

    return positions


positions = simulate(mass, G, length, initial_angle, 0.995, 2000)

plt.plot(positions)
plt.xlabel("Time")
plt.ylabel("Angle")
plt.axhline(0, color="red")
plt.show()
