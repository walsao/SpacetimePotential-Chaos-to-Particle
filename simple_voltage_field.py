import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Grid setup
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Voltage field - single Gaussian peak
A = 5
sigma = 1.5
V = A * np.exp(-((X**2 + Y**2) / (2 * sigma**2)))

# Electric field = -∇V
Ey, Ex = np.gradient(-V)

# Plot voltage surface
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, V, cmap=cm.viridis)
ax1.set_title("SpacetimePotential: Single Voltage Well")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("Voltage")

# Plot electric field vectors
ax2 = fig.add_subplot(1, 2, 2)
ax2.contourf(X, Y, V, cmap=cm.viridis, alpha=0.7)
ax2.quiver(X[::5, ::5], Y[::5, ::5], Ex[::5, ::5], Ey[::5, ::5], color='black')
ax2.set_title("Electric Field (−∇V)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.tight_layout()
plt.show()
