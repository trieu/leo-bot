#!/usr/bin/env bash

# =============================================================================
# Airflow 2.11.0 Development Environment Starter
#
# Description:
#   This script sets up and runs a local Apache Airflow development
#   environment. It installs a specific version of Airflow, initializes
#   the database, creates a default admin user, and starts the
#   webserver and scheduler.
#
# Updates:
#   - Skips installation if 'airflow' command is already available.
#   - Sets the current directory's 'dags' folder as AIRFLOW__CORE__DAGS_FOLDER.
#   - **Improved start/stop using TRAP for clean shutdown (SIGINT/SIGTERM/EXIT).**
#
# Author: Gemini
# Date: October 18, 2025
# =============================================================================

# --- Configuration ---
# Use a specific version to ensure a consistent environment.
AIRFLOW_VERSION="2.11.0"
# Keep all Airflow files (dags, logs, configs) in a dedicated directory.
export AIRFLOW_HOME="${HOME}/airflow_dev"
# Use a constraint file for reproducible and conflict-free installations.
PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
# Disable example DAGs for a cleaner UI.
export AIRFLOW__CORE__LOAD_EXAMPLES=False
# Set a default user for the initial setup.
ADMIN_USERNAME="admin"
ADMIN_PASSWORD="admin"
# Directory where the script is run, used for relative dags folder.
CURRENT_DIR="$(pwd)"

# Variables to store PIDs for clean shutdown
WEBSERVER_PID=""
SCHEDULER_PID=""


# --- Helper Functions ---
# Print a formatted message.
function print_message() {
    echo ""
    echo "================================================================="
    echo "  $1"
    echo "================================================================="
    echo ""
}

# Check if a command exists.
function command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Cleanup function to be executed on script exit (via trap).
function cleanup() {
    print_message "Stopping Airflow Services"
    local exit_status=$?

    if [ -n "$WEBSERVER_PID" ] && kill -0 "$WEBSERVER_PID" 2>/dev/null; then
        echo "-> Stopping Airflow Webserver (PID: ${WEBSERVER_PID})..."
        kill "$WEBSERVER_PID" 2>/dev/null
    else
        echo "-> Webserver PID is not set or process already stopped."
    fi

    if [ -n "$SCHEDULER_PID" ] && kill -0 "$SCHEDULER_PID" 2>/dev/null; then
        echo "-> Stopping Airflow Scheduler (PID: ${SCHEDULER_PID})..."
        kill "$SCHEDULER_PID" 2>/dev/null
    else
        echo "-> Scheduler PID is not set or process already stopped."
    fi

    # Wait for background jobs to actually terminate
    wait "$WEBSERVER_PID" "$SCHEDULER_PID" 2>/dev/null

    echo "-> Airflow services stopped."
    # Exit with the status of the process that triggered the trap
    exit $exit_status
}

# --- Main Script ---
# Set up a trap to call the cleanup function on script exit signals.
# This handles Ctrl+C (SIGINT), graceful termination (SIGTERM), and general exit.
trap cleanup SIGINT SIGTERM EXIT

set -e # Exit immediately if a command exits with a non-zero status.

print_message "Starting Airflow ${AIRFLOW_VERSION} Dev Setup"

# 1. Prerequisite Checks
if ! command_exists python3; then
    echo "[ERROR] python3 is not installed. Please install Python 3.8+ and try again."
    # Unset trap on failure checks to allow clean exit on error
    trap - SIGINT SIGTERM EXIT
    exit 1
fi
if ! command_exists pip3; then
    echo "[ERROR] pip3 is not installed. Please install pip3 and try again."
    trap - SIGINT SIGTERM EXIT
    exit 1
fi

echo "-> Python version ${PYTHON_VERSION} detected."
echo "-> Airflow will be installed in: ${AIRFLOW_HOME}"
mkdir -p "${AIRFLOW_HOME}" # Ensure the directory exists.


# 2. Airflow Installation Check & Install
if command_exists airflow; then
    echo "-> 'airflow' command detected. **Skipping Airflow installation.**"
else
    print_message "Installing apache-airflow==${AIRFLOW_VERSION}"
    pip3 install \
        "apache-airflow==${AIRFLOW_VERSION}" \
        --constraint "${CONSTRAINT_URL}"
fi


# 3. Configure DAGs Folder
# Check if a dags folder exists in the current directory and configure it.
if [ -d "${CURRENT_DIR}/dags" ]; then
    export AIRFLOW__CORE__DAGS_FOLDER="${CURRENT_DIR}/dags"
    echo "-> Found **${CURRENT_DIR}/dags**. Setting this as the Airflow DAGs folder."
else
    # Fallback to the default (or what Airflow will determine) and create a placeholder.
    DAGS_FOLDER_FALLBACK="${AIRFLOW_HOME}/dags"
    echo "-> **No 'dags' folder found in ${CURRENT_DIR}**. Using default: ${DAGS_FOLDER_FALLBACK}"
    mkdir -p "${DAGS_FOLDER_FALLBACK}"
    export AIRFLOW__CORE__DAGS_FOLDER="${DAGS_FOLDER_FALLBACK}"
fi


# 4. Initialize Database
print_message "Initializing Airflow database (airflow db init)"
airflow db init


# 5. Create Admin User
# Check if the user already exists to make the script re-runnable.
if ! airflow users list | grep -q "${ADMIN_USERNAME}"; then
    print_message "Creating admin user"
    airflow users create \
        --username "${ADMIN_USERNAME}" \
        --password "${ADMIN_PASSWORD}" \
        --firstname "Admin" \
        --lastname "User" \
        --role "Admin" \
        --email "admin@example.com"
else
    echo "-> Admin user '${ADMIN_USERNAME}' already exists. Skipping creation."
fi


# 6. Start Airflow Services
print_message "Starting Airflow Webserver and Scheduler"
# Start the webserver in the background on port 8080
airflow webserver --port 8080 &
WEBSERVER_PID=$!
echo "-> Webserver started in the background (PID: ${WEBSERVER_PID})"

# Start the scheduler in the background
airflow scheduler &
SCHEDULER_PID=$!
echo "-> Scheduler started in the background (PID: ${SCHEDULER_PID})"


# --- Final Instructions ---
print_message "Airflow Setup Complete!"
echo "Airflow UI is now available at: http://localhost:8080"
echo "Your DAGs folder is set to: **${AIRFLOW__CORE__DAGS_FOLDER}**"
echo ""
echo "  Login with:"
echo "    Username: ${ADMIN_USERNAME}"
echo "    Password: ${ADMIN_PASSWORD}"
echo ""
echo "Press **Ctrl+C** to stop the services and exit gracefully."
echo ""

# Keep the main script running indefinitely until interrupted
# This ensures the trap catches signals directed at the script, not the background processes
wait -n

# The wait command exits when the first background job terminates.
# We then rely on the trap to call cleanup() and stop the other job.