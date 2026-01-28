import subprocess

def launch_carla_server():
    carla_path = "/home/abdulrahman/CARLA_0.9.16/CarlaUE4.sh"  # Change if different!
    try:
        # call([carla_path, '-prefernvidia'])
        subprocess.Popen([carla_path, '-prefernvidia'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        msg = ">> CARLA server launched successfully."
    except Exception as e:
        msg = f">> Error launching CARLA: {e}"
    return msg


def close_carla_server():
    """Terminate the running CARLA server process if it exists."""
    # Try common CARLA process names to cover different launch setups.
    carla_process_names = ("CarlaUE4", "UE4-Linux-Shipping")
    try:
        for process_name in carla_process_names:
            result = subprocess.run(
                ["pkill", "-f", process_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return ">> CARLA server closed successfully."
        return ">> No running CARLA server processes were found."
    except Exception as e:
        return f">> Error closing CARLA: {e}"
