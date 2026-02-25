# Setting Up the Python Environment

## 1. Create a Virtual Environment (Recommended)

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 3. ROS Dependencies

- Make sure ROS (Melodic/Noetic) is installed and sourced.
- Build the ROS workspace:
```bash
cd ros_ws
catkin_make
source devel/setup.bash
```

## 4. Additional Notes
- For CoppeliaSim, follow instructions in `coppeliasim/README.md`.
- For MATLAB, see `mathematics/vision_matlab.m`.

---
This setup ensures full reproducibility and isolation of Python dependencies for research and development.
