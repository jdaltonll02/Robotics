import os
import subprocess
import yaml
import time
import logging
from datetime import datetime

EXPERIMENTS_DIR = "experiments"
CONFIG_PATH = "config.yaml"
LOGS_DIR = os.path.join(EXPERIMENTS_DIR, "logs")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(filename=os.path.join(LOGS_DIR, "experiment_runner.log"), level=logging.INFO)

def run_experiment(config_path=CONFIG_PATH, seed=0, ablation=None, baseline=None):
    """
    Runs a full closed-loop experiment with the given config, seed, and ablation/baseline options.
    Launches all ROS nodes, logs metrics, and saves results.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_seed{seed}"
    if ablation:
        exp_name += f"_ablation_{ablation}"
    if baseline:
        exp_name += f"_baseline_{baseline}"
    result_path = os.path.join(RESULTS_DIR, f"{exp_name}_{timestamp}.yaml")
    log_path = os.path.join(LOGS_DIR, f"{exp_name}_{timestamp}.log")
    logging.info(f"Starting experiment: {exp_name}")
    # Set random seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Launch ROS core and nodes (perception, planning, control)
    rosbag_path = os.path.join(RESULTS_DIR, f"{exp_name}_{timestamp}.bag")
    try:
        # Start rosbag record
        rosbag_proc = subprocess.Popen([
            "rosbag", "record", "-O", rosbag_path,
            "/camera/image_raw", "/segmentation", "/occupancy_grid", "/planned_path", "/cmd_vel"
        ])
        # Launch system (assumes launch file starts all nodes)
        launch_proc = subprocess.Popen([
            "roslaunch", "ros_ws/launch/full_system.launch"
        ], stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        # Wait for experiment duration
        duration = config.get('experiment', {}).get('duration', 60)
        time.sleep(duration)
        # Terminate processes
        launch_proc.terminate()
        rosbag_proc.terminate()
        logging.info(f"Experiment {exp_name} completed.")
    except Exception as e:
        logging.error(f"Experiment {exp_name} failed: {e}")
    # TODO: Add metrics extraction and YAML result saving
    # Example metrics: success_rate, path_efficiency, collision_rate, control_smoothness
    # Save metrics to result_path
    metrics = {
        "success_rate": None,
        "path_efficiency": None,
        "collision_rate": None,
        "control_smoothness": None,
        "seed": seed,
        "ablation": ablation,
        "baseline": baseline,
        "timestamp": timestamp
    }
    with open(result_path, 'w') as f:
        yaml.dump(metrics, f)
    logging.info(f"Results saved to {result_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a closed-loop robotics experiment.")
    parser.add_argument('--config', type=str, default=CONFIG_PATH)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ablation', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    args = parser.parse_args()
    run_experiment(args.config, args.seed, args.ablation, args.baseline)
