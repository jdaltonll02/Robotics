# Building a Differential-Drive Robot in CoppeliaSim

## Step-by-Step Instructions

### 1. Start CoppeliaSim
- Open CoppeliaSim.

### 2. Create the Robot Base
- Go to `Add` > `Primitive shape` > `Cuboid`.
- Set the size (e.g., X: 0.3m, Y: 0.2m, Z: 0.05m).
- Rename it to `robot_base`.

### 3. Add Wheels
- Add two cylinders: `Add` > `Primitive shape` > `Cylinder`.
- Set size (e.g., Diameter: 0.08m, Height: 0.03m).
- Rename them `left_wheel` and `right_wheel`.
- Position them on either side of the base (e.g., Y = ±0.12m).
- Set their orientation so the axis aligns with the robot’s Y-axis.

### 4. Attach Wheels to Base
- Select a wheel, then the base.
- Go to `Add` > `Joint` > `Revolute`.
- Set joint mode to `Velocity` and enable `Motor`.
- Attach each wheel to the base using a revolute joint.

### 5. Add a Caster Wheel (optional)
- Add a small sphere or cylinder at the rear for stability.

### 6. Add Camera Sensors
- Go to `Add` > `Vision sensor`.
- Place one or more vision sensors on the front/top of the base.
- Adjust the orientation and field of view as needed.
- Rename as `front_camera`, etc.

### 7. Group Components
- Select all parts (base, wheels, joints, cameras).
- Go to `Edit` > `Group` to make them a single model.

### 8. Set Model Properties
- Select the grouped model.
- Go to `File` > `Make selection a model`.
- Save the model: `File` > `Export selected models...` (save as `.ttm` in `coppeliasim/`).

### 9. Create a Scene
- Add obstacles or environment as needed.
- Save the scene: `File` > `Save scene as...` (save as `.ttt` in `coppeliasim/`).

### 10. Enable ROS Interface (for later)
- Make sure the `simExtROSInterface` plugin is enabled (check in `Help` > `About` > `Add-ons`).

---

**Tip:** You can adjust sizes, masses, and colors in the object properties dialog for each part.

Once your robot is built and saved, you can proceed to ROS integration and simulation experiments.
