#!/bin/bash

# --- Configuration and Setup ---

# FIXED BUG: Tilde (~) expansion inside quotes is unreliable. 
# We now explicitly use $HOME for a reliably resolved path.
# Define the Airflow Home directory. Defaults to $HOME/airflow_dev if AIRFLOW_HOME is not set.
AIRFLOW_HOME=${AIRFLOW_HOME:-"$HOME/airflow_dev"}
PID_DIR="$AIRFLOW_HOME"

echo "Attempting to stop Airflow services in PID directory: $PID_DIR"
echo "--------------------------------------------------------"

# Function to safely kill a process by PID stored in a file
stop_process_by_pid_file() {
    local component_name=$1
    # Check for the resolved path, not the literal string, by reading the var directly
    local pid_file="$PID_DIR/airflow-$component_name.pid"
    local PID # Declare PID locally for safety
    
    # BUG FIX: Renamed from PPID to PARENT_PID to avoid conflict with the shell's built-in read-only $PPID variable.
    local PARENT_PID

    echo "Checking for $component_name PID file: $pid_file"

    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        echo "Found $component_name running with PID $PID."
        
        # --- SCHEDULER LOGIC ---
        if [ "$component_name" == "scheduler" ]; then
            echo "Stopping scheduler (PID $PID)..."
            kill "$PID"
            
            # Check if the process received the signal successfully (kill returns 0 if signal sent)
            if [ $? -eq 0 ]; then
                echo "Signal sent successfully. Removing PID file."
                rm -f "$pid_file"
                echo "Scheduler stopped and PID file removed."
            else
                echo "Warning: Signal failed (PID $PID). Process may have already terminated."
                # We attempt to clean up the PID file regardless in case the process died ungracefully
                rm -f "$pid_file"
                echo "PID file cleaned up."
            fi
        
        # --- WEBSERVER LOGIC ---
        elif [ "$component_name" == "webserver" ]; then
            # Get the Parent PID (PARENT_PID) of the PID in the file (which is usually the Gunicorn worker)
            PARENT_PID=$(ps -o ppid= -p "$PID" 2>/dev/null | tr -d ' ')
            
            # 1. Kill the Gunicorn worker PID first
            echo "Stopping webserver worker (PID $PID)..."
            kill "$PID" 2>/dev/null
            
            # 2. Kill the parent process (the actual 'airflow webserver' command)
            if [ -n "$PARENT_PID" ] && ps -p "$PARENT_PID" > /dev/null; then
                echo "Stopping parent Airflow webserver process (PPID $PARENT_PID)..."
                kill "$PARENT_PID" 2>/dev/null
            elif [ -n "$PARENT_PID" ]; then
                echo "Warning: Parent process $PARENT_PID for webserver not found or already terminated."
            fi
            
            # 3. Clean up the PID file after attempting to signal both PIDs
            if [ -f "$pid_file" ]; then
                rm -f "$pid_file"
                echo "Webserver shutdown signals sent and PID file removed."
            else
                echo "Warning: Webserver PID file was unexpectedly missing after kill attempt."
            fi
        fi

    else
        echo "$component_name PID file not found. Assuming it is not running."
    fi

    echo "" # Newline for clean separation
}

# --- Execution ---

# Stop the Scheduler
stop_process_by_pid_file "scheduler"

# Stop the Webserver
stop_process_by_pid_file "webserver"

echo "Airflow shutdown attempt complete."

# Note: If processes are managed by systemd or Docker/Kubernetes, you must use
# the respective commands (e.g., 'sudo systemctl stop airflow-scheduler') instead.
