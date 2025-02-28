# Launch Experiments on TPU Pods with a Python interface

## Supports automatic management of spot (preemptible) pods
### (Makes them 10x more effective)

## Installation

### Google cloud setup
One way to do this (roughly) is:
```bash
# Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install

# Initialize and authenticate:
gcloud init

# Set your project:
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs:
gcloud services enable compute.googleapis.com tpu.googleapis.com

# Authenticate application default credentials:
gcloud auth application-default login
```
You know you did it right when you can run a command like this:
```
gcloud compute tpus tpu-vm create your-tpu-name --zone=us-central2-b --accelerator-type=v4-8 --preemptible --version=tpu-vm-v4-base
```


### TPU pod launcher setup
```
pip install git+https://github.com/Sea-Snell/tpu_pod_launcher.git@main
```
Then
```
cd tpu_pod_launcher
```
Then you should run 
```
python -m pip install -e .
```
Then modify launch.py so it works for you (last part).

### Modifying launch.py

Fill in the arguments in here
```
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
```
And fill in the TPUs you're working with (or will work with) here. Put their region in too:
```
    available_tpus = [
        ('some-v3-pod-in-europe', 'europe-west4-a'),
        ('some-other-v3-pod-in-europe', 'europe-west4-a'),
        ('some-v4-pod-in-us', 'us-central2-b'), 
        ('some-other-v4-pod-in-us', 'us-central2-b'),
    ]
```

## Useful info

tpu_pod_launcher.py has the core stuff - usually I don't change it that much.

launch.py is basically an interface for tpu_pod_launcher.py - I change it often - mostly updating the available_tpus part.

## Run_spot_scripts_auto.py

If you want this to work properly, it should be done on a machine that is always running. That means you shouldn't run it directly on your personal laptop or desktop since the script will stop as soon as your computer turns off. Ideally, you'll have a server/cluster that you can ssh into and run this from - luckily, if you dont have one, theres a workaround.

### The workaround

Create an small on-demand pod in any region, like a v2-8 or v3-8. I believe anyone in TRC can make one in us-central1-f for free. Then you can ssh into the host of that pod (its first external IP address) and start the installation process. The only caveat to this workaround is that on-demand pods go down from time to time (once every 1-2 months), which will stop the script. So always have your code backed up somewhere else.

### Effective usage

If you want completely automated training/inference/whatever on spot pods, the scripts being run on them need to accomodate for that. For example, if you're doing inference, your script should detect whether the output file for the data was already written to (which means your pod was preempted). If it was already written to, you need a mechanism that makes your inference script skip to wherever it left off.