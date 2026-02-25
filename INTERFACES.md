
## RL Control Interface Contract

- **ROS Topic:** `/rl_state` (custom message: rl_state.msg)
- **Published By:** World Representation/Planning Node
- **Consumed By:** RL Control Node

### Data Contract
- **Message:** `float32[] state_vector`
- **Shape:** `[N]` (configurable, e.g., N=100)
- **Frame:** `base_link` or `map`
- **Rate:** 10 Hz (configurable)
- **Action Space:** Continuous `[linear_vel, angular_vel]` (scaled by config)
- **Reward Function:** Goal reaching, smooth motion, collision avoidance (see RL code and /mathematics/rl_optimization.md)
- **Termination:** Collision, timeout, goal reached

### Example YAML Config
```yaml
rl_control:
  state_dim: 100
  action_dim: 2
  action_scale: [1.0, 1.0]
  rate: 10
```

### Failure Handling
- Log state vector shape mismatches
- Log and handle RL instability (NaN, divergence)

### References
- See `/mathematics/rl_optimization.md` for policy gradient and reward shaping equations.
# Perception-to-Planning Interface Contract

## ROS Topic: `/occupancy_grid`

- **Message Type:** `nav_msgs/OccupancyGrid`
- **Published By:** Perception Node
- **Consumed By:** Planning Node (A*, Hybrid A*)

### Data Contract
- **Shape:** `[height, width]` (matches simulation map)
- **Frame:** `map` (preferred) or `odom`
- **Resolution:** meters/pixel (configurable, e.g., 0.05)
- **Update Rate:** 10 Hz (configurable)
- **Data Encoding:**
  - `0`: Free space
  - `100`: Occupied
  - `-1`: Unknown
- **Source:** Derived from segmentation output (CNN)
- **Uncertainty:** Not encoded in this message; can be extended with a custom message if needed.

### Example YAML Config
```yaml
occupancy_grid:
  topic: /occupancy_grid
  frame_id: map
  resolution: 0.05
  rate: 10
  height: 128
  width: 128
```

### Failure Handling
- If segmentation output is missing or invalid, publish an empty grid with all values `-1` and log a warning.
- Log all contract violations (shape, frame, rate) for analysis.

### References
- See `/mathematics/kinematics_dynamics.md` for coordinate frame conventions.
- See `/mathematics/perception_loss.md` for segmentation loss equations.

---

This contract must be enforced in both perception and planning nodes. All changes must be documented and justified in code comments and documentation.
