# europe-west4-b
gcloud compute tpus queued-resources create tpu1 --node-id tpunode1 --project raden-tpu --zone europe-west4-b --accelerator-type v5litepod-4 --runtime-version v2-alpha-tpuv5-lite --spot
gcloud compute tpus queued-resources list --project raden-tpu --zone europe-west4-b
gcloud compute tpus queued-resources describe tpu1 --project raden-tpu --zone europe-west4-b
gcloud compute tpus tpu-vm describe tpunode1 --project raden-tpu --zone=europe-west4-b
34.34.119.30
gcloud compute tpus queued-resources delete tpu1 --project raden-tpu  --zone europe-west4-a --force --async
gcloud compute tpus queued-resources ssh tpu1 --project raden-tpu --zone europe-west4-a
gcloud compute instances start tpu16 --project raden-tpu --zone=europe-west4-a
ssh -R 34.34.119.30
ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.34.119.30
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" image_lagcodec/ muaz@34.34.119.30:~/qcute/image_lagcodec/
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" . muaz@34.34.119.30:~/qcute/
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split train --out_dir /dev/shm/imagenet64
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split validation --out_dir /dev/shm/imagenet64

# us-east1-d

gcloud compute tpus queued-resources create tpu1 --node-id tpunode1 --project raden-tpu --zone us-east1-d --accelerator-type v6e-8 --runtime-version v2-alpha-tpuv6e --spot
gcloud compute tpus queued-resources list --project raden-tpu --zone us-east1-d
gcloud compute tpus queued-resources describe tpu1 --project raden-tpu --zone us-east1-d
gcloud compute tpus tpu-vm describe tpunode1 --project raden-tpu --zone=us-east1-d
34.34.119.30
gcloud compute tpus queued-resources delete tpu1 --project raden-tpu  --zone us-east1-d --force --async
gcloud compute tpus queued-resources ssh tpu1 --project raden-tpu --zone us-east1-d
gcloud compute instances start tpu16 --project raden-tpu --zone=us-east1-d
ssh -R 34.34.119.30
ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.34.119.30
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" image_lagcodec/ muaz@34.34.119.30:~/qcute/image_lagcodec/
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" . muaz@34.34.119.30:~/qcute/
curl -LsSf https://astral.sh/uv/install.sh | sh
time uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split train --out_dir /dev/shm/imagenet64
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split validation --out_dir /dev/shm/imagenet64

# europe-west4-a

gcloud compute tpus queued-resources create tpu1 --node-id tpunode1 --project raden-tpu --zone europe-west4-a --accelerator-type v6e-8 --runtime-version v2-alpha-tpuv6e --spot
gcloud compute tpus queued-resources list --project raden-tpu --zone europe-west4-a
gcloud compute tpus queued-resources describe tpu1 --project raden-tpu --zone europe-west4-a
gcloud compute tpus tpu-vm describe tpunode1 --project raden-tpu --zone=europe-west4-a
34.158.147.98
gcloud compute tpus queued-resources delete tpu1 --project raden-tpu  --zone europe-west4-a --force --async
gcloud compute tpus queued-resources ssh tpu1 --project raden-tpu --zone europe-west4-a
ssh -R 34.158.147.98
ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.158.147.98
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" image_lagcodec/ muaz@34.158.147.98:~/qcute/image_lagcodec/
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" . muaz@34.158.147.98:~/qcute/
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split train --out_dir /dev/shm/imagenet64
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split validation --out_dir /dev/shm/imagenet64

##

gcloud compute tpus queued-resources ssh tpu16 --project raden-tpu --zone europe-west4-a --worker=0 --command="echo ok"
gcloud compute tpus queued-resources ssh tpu16 --project raden-tpu --zone europe-west4-a --worker=1 --command="echo ok"
gcloud compute tpus queued-resources ssh tpu16 --project raden-tpu --zone europe-west4-a --worker=2 --command="echo ok"
gcloud compute tpus queued-resources ssh tpu16 --project raden-tpu --zone europe-west4-a --worker=3 --command="echo ok"
gcloud compute tpus tpu-vm describe tpunode16 --project raden-tpu --zone=europe-west4-a 
#--format="value(networkEndpoints[0].accessConfig.externalIp)"
34.34.119.30
34.6.3.243
34.6.41.209
34.6.231.61

ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.34.119.30
ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.6.3.243
ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.6.41.209
ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.6.231.61
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" . muaz@34.6.231.61:~/qcute/

rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" image_lagcodec/ muaz@34.34.119.30:~/qcute/image_lagcodec/

rsync -avz --exclude="checkpoints/" -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu34-%r@%h:%p -i ~/.ssh/google_compute_engine" muaz@34.34.119.30:~/qcute/image_lagcodec/logs/imagenet64_par1 image_lagcodec/logs/

curl -LsSf https://astral.sh/uv/install.sh | sh
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split train --out_dir /dev/shm/imagenet64
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split validation --out_dir /dev/shm/imagenet64


#####

gcloud compute tpus queued-resources create tpu16 --node-id tpunode16 --project raden-tpu --zone europe-west4-a --accelerator-type v6e-16 --runtime-version v2-alpha-tpuv6e --spot
gcloud compute tpus queued-resources list --project raden-tpu --zone europe-west4-a
gcloud compute tpus queued-resources describe tpu16 --project raden-tpu --zone europe-west4-a
gcloud compute tpus queued-resources delete tpu16 --project raden-tpu  --zone europe-west4-a --force --async
gcloud compute tpus queued-resources ssh tpunode16 --project raden-tpu --zone europe-west4-a
gcloud compute instances start tpu16 --project raden-tpu --zone=europe-west4-a
#

gcloud compute tpus queued-resources ssh tpu16 --project raden-tpu --zone europe-west4-a --worker=0 --command="echo ok"
gcloud compute tpus queued-resources ssh tpu16 --project raden-tpu --zone europe-west4-a --worker=1 --command="echo ok"
gcloud compute tpus queued-resources ssh tpu16 --project raden-tpu --zone europe-west4-a --worker=2 --command="echo ok"
gcloud compute tpus queued-resources ssh tpu16 --project raden-tpu --zone europe-west4-a --worker=3 --command="echo ok"
gcloud compute tpus tpu-vm describe tpunode16 --project raden-tpu --zone=europe-west4-a 
#--format="value(networkEndpoints[0].accessConfig.externalIp)"
34.34.119.30
34.6.3.243
34.6.41.209
34.6.231.61

ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.34.119.30
ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.6.3.243
ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.6.41.209
ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.6.231.61
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" . muaz@34.6.231.61:~/qcute/

rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" image_lagcodec/ muaz@34.34.119.30:~/qcute/image_lagcodec/

rsync -avz --exclude="checkpoints/" -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu34-%r@%h:%p -i ~/.ssh/google_compute_engine" muaz@34.34.119.30:~/qcute/image_lagcodec/logs/imagenet64_par1 image_lagcodec/logs/

curl -LsSf https://astral.sh/uv/install.sh | sh
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split train --out_dir /dev/shm/imagenet64
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split validation --out_dir /dev/shm/imagenet64
