import textwrap
import os
from tpu_pod_launcher import TPUPodClient, TPUPodProject, create_cli
import time
import threading
import subprocess
import shlex

SETUP_SCRIPT = """\
cd ~/
# install basics
apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    curl \
    git \
    vim \
    wget \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniforge
rm -rf ~/Miniconda3-py39_4.12.0-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -P ~/
bash ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b

# install dependencies
source ~/miniconda3/bin/activate
conda init bash
conda create -n python-for-my-training-repo python=3.10 -y
conda activate python-for-my-training-repo
cd ~/my-training-repo

# assuming you have a setup.py in my-training-repo - if not, set up however you usually do it
python -m pip install -e .

# install jax, not necessarily required, depends on your project
python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# clean up
cd ~/
rm -rf ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
""".strip()

CHECK_DEVICES = r"""
source ~/miniconda3/bin/activate
python -c "import jax; print(jax.devices())"
""".strip()

def setup(project: TPUPodProject, verbose: bool=False):
    project.copy(verbose=verbose)
    project.ssh(SETUP_SCRIPT, verbose=verbose)
    project.ssh('mkdir ~/.config/', verbose=verbose)
    project.ssh('mkdir ~/.config/gcloud/', verbose=verbose)
    project.scp('/home/mark/.config/gcloud/my_gcs_key.json', '~/.config/gcloud/', verbose=verbose)

def check_devices(project: TPUPodProject, verbose: bool=False):
    project.ssh(CHECK_DEVICES, verbose=verbose)

def debug(project: TPUPodProject, verbose: bool=False):
    import IPython; IPython.embed()

def create_tpu(project: TPUPodProject, accelerator_type: str='v4-8', software_version: str='tpu-vm-v4-base', **kwargs):
    project.client.create(
        tpu_name=project.tpu_name,
        accelerator_type=accelerator_type,
        software_version=software_version,
        **kwargs,
    )

# doesnt actually create_tpu_until_fail, just tries creating it once. but thats fine.
def create_tpu_until_fail(project: TPUPodProject, accelerator_type: str='v3-256', software_version: str='tpu-ubuntu2204-base', **kwargs):
    while True:
        try:
            create_tpu(project, accelerator_type, software_version, **kwargs)
            print("TPU created successfully.")
            break
        except subprocess.CalledProcessError as e:
            print(f"Failed to create TPU: {e.output}. Retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying

def create_spot_tpu(project: TPUPodProject, accelerator_type: str='v4-32', software_version: str='tpu-vm-v4-base', **kwargs):
    project.client.create_spot(
        tpu_name=project.tpu_name,
        accelerator_type=accelerator_type,
        software_version=software_version,
        **kwargs,
    )

# doesnt actually create_tpu_until_fail, just tries creating it once. but thats fine.
def create_spot_tpu_until_fail(project: TPUPodProject, accelerator_type: str, software_version: str=None, **kwargs):
    # Automatically set the software version based on the accelerator type
    if software_version is None:
        if 'v3' in accelerator_type:
            software_version = 'tpu-ubuntu2204-base'
        elif 'v4' in accelerator_type:
            software_version = 'tpu-vm-v4-base'
        elif 'v5e' in accelerator_type:
            software_version = 'v2-alpha-tpuv5-lite'
    
    while True:
        try:
            create_spot_tpu(project, accelerator_type, software_version, **kwargs)
            print("TPU created successfully.")
            break
        except subprocess.CalledProcessError as e:
            print(f"Failed to create TPU: {e}. Retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying

def simulate_interruption(project: TPUPodProject, **kwargs):
    """
    Simulates an interruption by deleting the TPU.
    """
    print(f"Simulating interruption for TPU {project.tpu_name}...")
    project.client.delete(project.tpu_name, **kwargs)
    print(f"TPU {project.tpu_name} has been deleted.")


def create_project(tpu_name: str, zone: str) -> TPUPodProject:
    return TPUPodProject(
        client=TPUPodClient(
            tpu_project='my-project',
            tpu_zone=zone,
            user='my-username', 
            key_path='/home/mark/my-ssh-key', # e.g. '/home/mark/.ssh/id_rsa',
        ),
        tpu_name=tpu_name,
        # fill in your own dirs
        copy_dirs=[('/home/mark/my-training-repo', '~/my-training-repo'), ('/home/mark/maybe-another-repo', '~/maybe-another-repo')],
        working_dir='~/my-training-repo',
        copy_excludes=['.git', '__pycache__'],
        kill_commands=['pkill -9 python'],
        setup_fn=setup
    )

if __name__ == "__main__":
    launch_config_path = os.path.join(os.path.dirname(__file__), 'launch_config.json')
    # EXAMPLE TPUS - FILL IN YOURSELF
    # These dont need to exist to be here
    available_tpus = [
        ('some-v3-pod-in-europe', 'europe-west4-a'),
        ('some-other-v3-pod-in-europe', 'europe-west4-a'),
        ('some-v4-pod-in-us', 'us-central2-b'), 
        ('some-other-v4-pod-in-us', 'us-central2-b'),
    ]

    tpu_projects = {name: create_project(name, zone) for name, zone in available_tpus}

    create_cli(
        projects=tpu_projects,
        setup=setup,
        custom_commands={
            'check_devices': check_devices,
            'debug': debug,
            'create_tpu': create_tpu,
            'create_spot_tpu': create_spot_tpu,
            'create_tpu_r': create_tpu_until_fail,
            'create_spot_tpu_r': create_spot_tpu_until_fail,
            'destroy': simulate_interruption,
        },
        launch_config_path=launch_config_path,
    )