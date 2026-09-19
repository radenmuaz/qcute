# socket refresh
gcloud compute tpus tpu-vm ssh tpunode1 --project raden-tpu --zone us-central2-b --command="echo ok"
ssh -o ControlMaster=auto -o ControlPersist=4h -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -o StrictHostKeyChecking=accept-new -i ~/.ssh/google_compute_engine -fN muaz@35.186.98.243

gcloud compute tpus tpu-vm ssh tpunode2 --project raden-tpu --zone us-central2-b --command="echo ok"
ssh -o ControlMaster=auto -o ControlPersist=4h -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -o StrictHostKeyChecking=accept-new -i ~/.ssh/google_compute_engine -fN muaz@35.186.15.67

gcloud compute tpus tpu-vm ssh tpunode3 --project raden-tpu --zone us-central2-b --command="echo ok"
ssh -o ControlMaster=auto -o ControlPersist=4h -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -o StrictHostKeyChecking=accept-new -i ~/.ssh/google_compute_engine -fN muaz@107.167.160.20

gcloud compute tpus tpu-vm ssh tpunode4 --project raden-tpu --zone us-central2-b --command="echo ok"
ssh -o ControlMaster=auto -o ControlPersist=4h -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -o StrictHostKeyChecking=accept-new -i ~/.ssh/google_compute_engine -fN muaz@35.186.33.7

# ssh
ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.98.243
ssh -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.15.67
ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@107.167.160.20
ssh -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.33.7
ssh -o ControlPath=~/.ssh/controlmasters/tpu5-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.115.139
ssh -o ControlPath=~/.ssh/controlmasters/tpu6-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.86.22
ssh -o ControlPath=~/.ssh/controlmasters/tpu7-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.34.230
ssh -o ControlPath=~/.ssh/controlmasters/tpu8-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.110.50
# attach
tmux attach -t search1_base
tmux attach -t search2_lookahead
tmux attach -t search3_past
tmux attach -t search4_future
tmux attach -t search5_lookahead_past
tmux attach -t search6_lookahead_future
tmux attach -t search7_past_future
tmux attach -t search8_all

# pull lagcodec
rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@35.186.98.243:~/qcute/image_lagcodec/logs/<run_name>/ \
  image_lagcodec/logs/<run_name>/

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@35.186.15.67:~/qcute/image_lagcodec/logs/<run_name>/ \
  image_lagcodec/logs/<run_name>/

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@107.167.160.20:~/qcute/image_lagcodec/logs/<run_name>/ \
  image_lagcodec/logs/<run_name>/

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@35.186.33.7:~/qcute/image_lagcodec/logs/<run_name>/ \
  image_lagcodec/logs/<run_name>/

## pull example
rsync -avz --exclude="checkpoints/" -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" muaz@35.186.98.243:~/qcute/image_lagcodec/logs/1_baseline/ image_lagcodec/logs/

rsync -avz --exclude="checkpoints/" -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -i ~/.ssh/google_compute_engine" muaz@35.186.15.67:~/qcute/image_lagcodec/logs/run2 image_lagcodec/logs/

rsync -avz --exclude="checkpoints/" -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine" muaz@107.167.160.20:~/qcute/image_lagcodec/logs/run3 image_lagcodec/logs/

rsync -avz --exclude="checkpoints/" -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -i ~/.ssh/google_compute_engine" muaz@35.186.33.7:~/qcute/image_lagcodec/logs/run4 image_lagcodec/logs/

# push lagcodec
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" image_lagcodec/ muaz@35.186.98.243:~/qcute/image_lagcodec/

rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" image_lagcodec/ muaz@35.186.15.67:~/qcute/image_lagcodec/

rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" image_lagcodec/ muaz@107.167.160.20:~/qcute/image_lagcodec/

rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  --filter=':- ../.gitignore' --exclude=".git/" \
  /Users/muaz/code/qcute/image_lagcodec/ muaz@35.186.33.7:~/qcute/image_lagcodec/

# push reset all
rsync -avz --delete \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  --filter=':- .gitignore' --exclude=".git/" \
  /Users/muaz/code/qcute/ muaz@35.186.98.243:~/qcute/

rsync -avz --delete -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- .gitignore' --exclude=".git/" /Users/muaz/code/qcute/ muaz@35.186.15.67:~/qcute/

rsync -avz --delete -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- .gitignore' --exclude=".git/" /Users/muaz/code/qcute/ muaz@107.167.160.20:~/qcute/

rsync -avz --delete \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  --filter=':- .gitignore' --exclude=".git/" \
  /Users/muaz/code/qcute/ muaz@35.186.33.7:~/qcute/





## older
# pull sample
gcloud compute tpus queued-resources scp muaz@tpu1:/home/muaz/qcute/image_lagcodec/logs/cifar10_stack_lag0/samples_epoch20_reconstruct_val.png . --project raden-tpu --zone us-central2-b

# rsync config folder
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine" /Users/muaz/code/qcute/image_lagcodec/configs/ muaz@107.167.160.20:~/qcute/image_lagcodec/configs/

# ssh direct
ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.98.243
ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@107.167.160.20

# rsync lagcodec
rsync -avz --filter=":- /Users/muaz/code/qcute/.gitignore" --exclude=".git" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  /Users/muaz/code/qcute/image_lagcodec/ muaz@107.167.160.20:~/qcute/image_lagcodec/


# get ip
curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip"

gcloud compute tpus tpu-vm describe tpunode1 --project raden-tpu --zone us-central2-b --format="value(networkEndpoints[0].accessConfig.externalIp)" 2>&1

# update ssh
ssh -o ControlMaster=auto -o ControlPersist=4h -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -o StrictHostKeyChecking=accept-new -i ~/.ssh/google_compute_engine -fN muaz@35.186.98.243
ssh -o ControlMaster=auto -o ControlPersist=4h -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -o StrictHostKeyChecking=accept-new -i ~/.ssh/google_compute_engine -fN muaz@35.186.15.67
ssh -o ControlMaster=auto -o ControlPersist=4h -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -o StrictHostKeyChecking=accept-new -i ~/.ssh/google_compute_engine -fN muaz@107.167.160.20
echo "tpu2:"; ssh -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.15.67 "echo connected"
echo "tpu3:"; ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@107.167.160.20 "echo connected"

gcloud compute tpus tpu-vm ssh tpunode4 --project raden-tpu --zone us-central2-b --command="echo ok" 2>&1 | tail -5
ssh -o ControlMaster=auto -o ControlPersist=4h -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -o StrictHostKeyChecking=accept-new -i ~/.ssh/google_compute_engine -fN muaz@35.186.33.7
ssh -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.33.7 "echo connected"

# rsync 
rsync -avz \
  --exclude="__pycache__/" --exclude="*.pyc" --exclude="*.pyo" --exclude="*.egg-info/" --exclude=".venv/" --exclude=".pytest_cache/" --exclude=".ruff_cache/" --exclude=".mypy_cache/" --exclude="dist/" --exclude="build/" --exclude="datasets/" --exclude="logs/" --exclude="checkpoints/" --exclude=".env" --exclude=".git" -e \
  "ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  /Users/muaz/code/qcute/image_lagcodec/ muaz@107.167.160.20:~/qcute/image_lagcodec/

# rsync samples and logs

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@35.186.98.243:~/qcute/image_lagcodec/logs/cifar10_stack_fair1024_b/ \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_stack_fair1024_b/

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@35.186.15.67:~/qcute/image_lagcodec/logs/cifar10_stack_fair1024_c/ \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_stack_fair1024_c/

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@107.167.160.20:~/qcute/image_lagcodec/logs/cifar10_curr_off/ \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_curr_off/

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@35.186.33.7:~/qcute/image_lagcodec/logs/cifar10_curr_off_v16/ \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_curr_off_v16/

# another for lagcodec cifar full runs

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@35.186.98.243:~/qcute/image_lagcodec/logs/cifar10_stack_fair1024_b/ \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_stack_fair1024_b/

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@35.186.15.67:~/qcute/image_lagcodec/logs/cifar10_stack_fair1024_c/ \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_stack_fair1024_c/


# scp if rsync ssh not working

gcloud compute tpus queued-resources scp \
  'muaz@tpu1:~/qcute/image_lagcodec/logs/cifar10_stack_fair1024_b/samples_*.png' \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_stack_fair1024_b/ \
  --project raden-tpu --zone us-central2-b

gcloud compute tpus queued-resources scp \
  'muaz@tpu2:~/qcute/image_lagcodec/logs/cifar10_stack_fair1024_c/samples_*.png' \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_stack_fair1024_b/ \
  --project raden-tpu --zone us-central2-b

# more

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@107.167.160.20:~/qcute/image_lagcodec/logs/cifar10_curr_off_s4_gumbel001/ \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_curr_off_s4_gumbel001/ 2>&1 | tail -5

rsync -avz --exclude="checkpoints/" \
  -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -i ~/.ssh/google_compute_engine" \
  muaz@35.186.33.7:~/qcute/image_lagcodec/logs/cifar10_curr_off_s4_gumbel2/ \
  /Users/muaz/code/qcute/image_lagcodec/logs/cifar10_curr_off_s4_gumbel2/ 2>&1 | tail -5

# tpu5
gcloud compute tpus queued-resources ssh tpu5 --project raden-tpu --zone us-central2-b --command="echo ok"
rm -f ~/.ssh/controlmasters/tpu5-muaz@35.186.33.7:22
ssh -o ControlMaster=auto -o ControlPersist=600 \
  -o ControlPath=~/.ssh/controlmasters/tpu5-%r@%h:%p \
  -o StrictHostKeyChecking=accept-new \
  -i ~/.ssh/google_compute_engine muaz@35.186.33.7 "echo connected"

rm -f ~/.ssh/controlmasters/tpu5-muaz@35.186.33.7:22
ssh -o ControlMaster=auto -o ControlPersist=600 \
  -o ControlPath=~/.ssh/controlmasters/tpu5-%r@%h:%p \
  -i ~/.ssh/google_compute_engine muaz@35.186.33.7 "echo connected"

rsync -avz \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='logs' --exclude='checkpoints' --exclude='.venv' \
  --exclude='datasets' --exclude='*.pt' --exclude='*.ckpt' \
  --exclude='.env' \
  -e "ssh -o ControlPath=$HOME/.ssh/controlmasters/tpu5-%r@%h:%p -i $HOME/.ssh/google_compute_engine" \
  /Users/muaz/code/qcute/ muaz@35.186.110.50:~/qcute/

# tpu6
gcloud compute tpus queued-resources ssh tpu6 --project raden-tpu --zone us-central2-b --command="echo ok"
rm -f ~/.ssh/controlmasters/tpu6-muaz@35.186.110.50:22
ssh -o ControlMaster=auto -o ControlPersist=600 \
  -o ControlPath=~/.ssh/controlmasters/tpu6-%r@%h:%p \
  -o StrictHostKeyChecking=accept-new \
  -i ~/.ssh/google_compute_engine muaz@35.186.110.50 "echo connected"

rm -f ~/.ssh/controlmasters/tpu6-muaz@35.186.110.50:22
ssh -o ControlMaster=auto -o ControlPersist=600 \
  -o ControlPath=~/.ssh/controlmasters/tpu6-%r@%h:%p \
  -i ~/.ssh/google_compute_engine muaz@35.186.110.50 "echo connected"

rsync -avz \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='logs' --exclude='checkpoints' --exclude='.venv' \
  --exclude='datasets' --exclude='*.pt' --exclude='*.ckpt' \
  --exclude='.env' \
  -e "ssh -o ControlPath=$HOME/.ssh/controlmasters/tpu6-%r@%h:%p -i $HOME/.ssh/google_compute_engine" \
  /Users/muaz/code/qcute/ muaz@35.186.110.50:~/qcute/

# configs only
echo 'scp -o ControlPath=$HOME/.ssh/controlmasters/tpu5-%r@%h:%p -i $HOME/.ssh/google_compute_engine \
  /Users/muaz/code/qcute/summformer_jax/image_classification/configs/tiny_vit_like.py \
  muaz@35.186.33.7:~/qcute/summformer_jax/image_classification/configs/tiny_vit_like.py

scp -o ControlPath=$HOME/.ssh/controlmasters/tpu6-%r@%h:%p -i $HOME/.ssh/google_compute_engine \
  /Users/muaz/code/qcute/summformer_jax/image_classification/configs/tiny_vit_like.py \
  /Users/muaz/code/qcute/summformer_jax/image_classification/configs/tiny_vit_like_bidir_shared.py \
  muaz@35.186.110.50:~/qcute/summformer_jax/image_classification/configs/'