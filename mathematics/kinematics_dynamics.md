# Differential-Drive Robot Kinematics and Dynamics

## Kinematics

Let $v$ be linear velocity, $\omega$ angular velocity, $r$ wheel radius, $L$ wheelbase.

$$
\dot{x} = v \cos \theta \\
\dot{y} = v \sin \theta \\
\dot{\theta} = \omega
$$

Wheel velocities:
$$
v = \frac{r}{2}(\omega_R + \omega_L) \\
\omega = \frac{r}{L}(\omega_R - \omega_L)
$$

## Dynamics

Assume mass $m$, inertia $I$:
$$
F = m \dot{v} \\
\tau = I \dot{\omega}
$$
