import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid setup for the voltage field
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)

# Define voltage wells (spacetime curvature)
A1, sigma1, pos1 = 10, 1.0, (-3, -3)
A2, sigma2, pos2 = -6, 1.5, (4, 4)
V = (A1 * np.exp(-((X - pos1[0])**2 + (Y - pos1[1])**2) / (2 * sigma1**2)) +
     A2 * np.exp(-((X - pos2[0])**2 + (Y - pos2[1])**2) / (2 * sigma2**2)))

# Electric field = -∇V
Ey, Ex = np.gradient(-V, y, x)

# Particle setup
num_particles = 100
positions = np.random.uniform(-9, 9, (num_particles, 2))
velocities = np.zeros_like(positions)
dt = 0.05
mass = 1.0
charge = -1.0  # electron

# Interpolation helpers
x_vals = x
y_vals = y

def get_field_force(x, y):
    x = np.clip(x, x_vals[0], x_vals[-1])
    y = np.clip(y, y_vals[0], y_vals[-1])
    ix = np.searchsorted(x_vals, x) - 1
    iy = np.searchsorted(y_vals, y) - 1
    ix = np.clip(ix, 0, len(x_vals) - 2)
    iy = np.clip(iy, 0, len(y_vals) - 2)
    fx = Ex[iy, ix]
    fy = Ey[iy, ix]
    return fx, fy

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
field_plot = ax.contourf(X, Y, V, levels=200, cmap='plasma', alpha=0.6)
quiv = ax.quiver(X[::25, ::25], Y[::25, ::25], Ex[::25, ::25], Ey[::25, ::25], color='black')
scat = ax.scatter(positions[:, 0], positions[:, 1], c='cyan', s=10, label='Particles')
ax.legend()
ax.set_title("🧪 Chaos-to-Particle Simulation – SpacetimePotential")

def update(frame):
    global positions, velocities
    new_positions = []
    for i, (x, y) in enumerate(positions):
        fx, fy = get_field_force(x, y)
        ax_, ay_ = (charge * fx / mass), (charge * fy / mass)
        velocities[i] += np.array([ax_, ay_]) * dt
        positions[i] += velocities[i] * dt
        new_positions.append(positions[i])
    scat.set_offsets(new_positions)
    return scat,

ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)
plt.show()
