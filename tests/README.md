# Testing Pipeline

## 1. Python Unit Tests

- Run all Python tests:
  ```bash
  pytest tests/
  ```
- Example: Checks perception and RL model forward passes.

## 2. ROS Node Launch/Health Check

- Run ROS node integration test:
  ```bash
  cd tests
  bash test_ros_nodes.sh
  ```
- Verifies all core ROS nodes launch and are alive.

## 3. CI Integration (Optional)

- Add `pytest` and shell test scripts to your CI pipeline (e.g., GitHub Actions, GitLab CI).

---
This ensures both core algorithms and system integration are tested for every change.
