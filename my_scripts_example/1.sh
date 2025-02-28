source ~/miniconda3/bin/activate
conda activate python-for-my-training-repo

# generally everything set up here is useful to have:

# path to gcloud token that was copied to the pod
export GCLOUD_TOKEN_PATH='/home/mark/.config/gcloud/{your-key}.json'
# your huggingface token (in this exact format, no quotes)
export HF_TOKEN=my_hf_token
# your gcloud project name
export GCLOUD_PROJECT='my-project'
# your wandb api key
export WANDB_API_KEY='my-wandb-api-key'



# assuming train_script.py is in my-training-repo
python3 v2_llama_prm_train_automated.py \
    --train-data-path='path/to/train/dataset1' \
    --bsize=64 \
    --evalbsize=512 \
    --whatever=grok3_da_goat
