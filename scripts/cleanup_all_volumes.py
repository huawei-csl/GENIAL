import docker
import shutil
from pathlib import Path

# Initialize the Docker client
client = docker.from_env()


def remove_bound_directory(volume_name):
    # Get the volume object
    volume = client.volumes.get(volume_name)
    # Access the volume's mountpoint (which should be a temporary directory)
    mountpoint = Path(volume.attrs.get("Options")["device"])

    # Check if the volume is a bind mount by verifying the directory exists
    if mountpoint.exists() and mountpoint.parent.name == "tmp":
        print(f"Removing temporary directory: {mountpoint}")
        # Unmount and remove the directory
        shutil.rmtree(mountpoint, ignore_errors=True)
        print(f"Bound directory '{mountpoint}' has been removed.")


def find_active_volumes():
    # Get all existing volumes
    all_volumes = client.volumes.list()
    # Get only active (running) containers
    running_containers = client.containers.list()

    active_volumes = set()

    # Check each running container to see if it uses any of the volumes
    for container in running_containers:
        container_volumes = [mount["Name"] for mount in container.attrs["Mounts"] if "Name" in mount]
        active_volumes.update(container_volumes)

    # Display the status of each volume
    for volume in all_volumes:
        if "_exp_" in volume.name:
            if volume.name in active_volumes:
                print(f"Volume '{volume.name}' is in use by an active container.")
            else:
                print(
                    f"Volume '{volume.name}' is not in use by any active container, removing temporary directory and volume."
                )
                remove_bound_directory(volume.name)
                volume.remove()


# Run the function
if __name__ == "__main__":
    find_active_volumes()
