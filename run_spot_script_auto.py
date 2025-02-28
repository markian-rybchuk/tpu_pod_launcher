
import time
import json
import subprocess
from pprint import pprint
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Script mapping
# Format: 'some available tpu name': 'path of LOCAL script to run on the pod'
script_mapping = {
    'some-v3-pod-in-europe':  '/home/mark/tpu_pod_launcher/my_scripts_example/1.sh',
    'some-other-v3-pod-in-europe':  '/home/mark/tpu_pod_launcher/my_scripts_example/2.sh',
    'some-v4-pod-in-us':  '/home/mark/tpu_pod_launcher/my_scripts_example/3.sh',
    'some-other-v4-pod-in-us':  '/home/mark/tpu_pod_launcher/my_scripts_example/4.sh',
}

v3_256_projects = {
    'some-v3-pod-in-europe',
}
v3_128_projects = {
    'some-other-v3-pod-in-europe',
}
v4_64_projects = {
    'some-v4-pod-in-us', 'some-other-v4-pod-in-us',
}

def run_cmd(cmd):
    """Helper to run a command and print it for easier debugging."""
    print(f"[CMD] {cmd}")
    return subprocess.run(cmd, shell=True)

def create_setup_and_launch_tpu(project_name):
    """
    Create, setup, and launch a TPU instance *sequentially* for a single project.
    """
    print(f"=== Creating TPU for {project_name}...")
    if project_name in v3_256_projects:
        # format: python launch.py create_spot_tpu_r [accelerator_type] --project [your_gcloud_project_name]
        # automatically figures out the region and software version based on the accelerator type
        create_cmd = f"python launch.py create_spot_tpu_r v3-256 --project {project_name}"
    elif project_name in v3_128_projects:
        create_cmd = f"python launch.py create_spot_tpu_r v3-128 --project {project_name}"
    elif project_name in v4_64_projects:
        # since create_spot_tpu_until_fail doesn't really work right now, we just keep trying to create it until it works
        # creating a v3 pod always works first try for me so the code above is fine
        create_cmd = f"""while true; do
            if gcloud compute tpus tpu-vm create {project_name} \
                --zone=us-central2-b \
                --accelerator-type=v4-64 \
                --preemptible \
                --version=tpu-vm-v4-base; then
                break
            else
                sleep 60
            fi
        done"""
    else:
        raise ValueError(f"Unknown project name: {project_name}")
    
    run_cmd(create_cmd)
    
    print(f"=== Setting up TPU for {project_name}...")
    setup_cmd = f"python launch.py setup --project {project_name}"
    run_cmd(setup_cmd)
    
    print(f"=== Launching TPU for {project_name}...")
    script_path = script_mapping[project_name]
    launch_cmd = f"python launch.py launch {script_path} --project {project_name}"
    run_cmd(launch_cmd)

def destroy_tpu(project_name):
    """Destroy a TPU instance."""
    print(f"=== Destroying TPU for {project_name}...")
    destroy_cmd = f"python launch.py destroy --project {project_name}"
    run_cmd(destroy_cmd)

def recreate_tpu(project_name):
    """
    Destroy if it exists/not-ready, then create/setup/launch again.
    This function is intended to be called in parallel threads.
    """
    print(f"\nRecreating {project_name} (destroy + create/setup/launch).")
    destroy_tpu(project_name)
    create_setup_and_launch_tpu(project_name)

def main():
    """Monitor & manage TPUs with parallel creation/re-creation for missing/non-ready ones."""
    while True:
        try:
            # Collect data from both zones
            cmd_eu_a = "gcloud compute tpus tpu-vm list --zone=europe-west4-a --format=json"
            cmd_eu_b = "gcloud compute tpus tpu-vm list --zone=europe-west4-b --format=json"
            cmd_us = "gcloud compute tpus tpu-vm list --zone=us-central2-b --format=json"

            result1 = subprocess.run(cmd_eu_a, shell=True, capture_output=True, text=True)
            result2 = subprocess.run(cmd_us, shell=True, capture_output=True, text=True)
            result3 = subprocess.run(cmd_eu_b, shell=True, capture_output=True, text=True)
            
            tpu_data = []
            if result1.stdout:
                tpu_data.extend(json.loads(result1.stdout))
            if result2.stdout:
                tpu_data.extend(json.loads(result2.stdout))
            if result3.stdout:
                tpu_data.extend(json.loads(result3.stdout))
            
            # Filter for v3spot named TPUs across both zones
            v3spot_data = [
                vm for vm in tpu_data if
                vm.get('name', '').startswith('projects/prm-research/locations/europe-west4-a/nodes/v3spot')
            ]
            
            v4spot_data = [
                vm for vm in tpu_data if
                vm.get('name', '').startswith('projects/prm-research/locations/us-central2-b/nodes/v4spot')
            ]
            
            v5spot_data = [
                vm for vm in tpu_data if
                vm.get('name', '').startswith('projects/prm-research/locations/europe-west4-b/nodes/v5spot')
            ]
            
            existing_tpus = set(vm['name'].split('/')[-1] for vm in v3spot_data + v4spot_data + v5spot_data)
            
            # Build lists for parallel tasks:
            # 1) Non-ready => must recreate
            # 2) Missing => must create
            recreate_list = []
            create_list = []
            full_data = v3spot_data + v4spot_data + v5spot_data
            for vm in full_data:
                name = vm['name'].split('/')[-1]
                if vm['state'] != 'READY' and name in script_mapping:
                    recreate_list.append(name)

            for project_name in script_mapping.keys():
                if project_name not in existing_tpus:
                    create_list.append(project_name)
            
            # Run each group in parallel. Each project runs create_setup_and_launch or recreate in its own thread.
            # Adjust max_workers as needed for your environment.
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = []
                # Re-create any that are not READY
                for name in recreate_list:
                    futures.append(executor.submit(recreate_tpu, name))
                
                # Create missing ones
                for proj in create_list:
                    futures.append(executor.submit(create_setup_and_launch_tpu, proj))

                # Wait for them all to complete
                for fut in as_completed(futures):
                    # If any function raises an exception, re-raise it here.
                    _ = fut.result()
            
            # Print a summary
            print("\n=== TPU Instance Status ===")
            for vm in full_data:
                name = vm['name'].split('/')[-1]
                if name in script_mapping:
                    state = vm['state']
                    created = datetime.strptime(vm['createTime'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
                    print(f"TPU {name}:")
                    print(f"  State: {state}")
                    print(f"  Created: {created}")
                    print("---")
            print(f"\nTotal instances: {len(v3spot_data)}\n")
            
            time.sleep(600)
        
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
