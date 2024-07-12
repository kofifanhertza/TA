from subprocess import run, CalledProcessError
from time import sleep

# Path and name to the script you are trying to start
file_path = "detect_test_final_TA.py"

restart_timer = 30

def start_script():
    while True:
        try:
            # Run the script using subprocess.run
            run(["python", file_path], check=True)
        except CalledProcessError as e:
            # Handle the crash
            print(f"Script crashed with error: {e}. Restarting in {restart_timer} seconds...")
            sleep(restart_timer)  # Wait before restarting

# Start the script
start_script()