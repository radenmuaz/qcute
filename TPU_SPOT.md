gcloud compute tpus queued-resources create tpu64 --node-id tpunode64 --project raden-tpu --zone europe-west4-a --accelerator-type v6e-64 --runtime-version v2-alpha-tpuv6e --spot

gcloud compute tpus queued-resources create tpu32 --node-id tpunode32 --project raden-tpu --zone europe-west4-a --accelerator-type v6e-32 --runtime-version v2-alpha-tpuv6e --spot

gcloud compute tpus queued-resources describe tpu1 --project raden-tpu --zone europe-west4-a

gcloud compute tpus queued-resources delete tpu8 --project raden-tpu  --zone europe-west4-a --force --async

gcloud compute tpus queued-resources ssh tpu1--project raden-tpu --zone europe-west4-a

#

gcloud compute tpus queued-resources ssh tpu16 --project raden-tpu --zone europe-west4-a --command="echo ok"
gcloud compute tpus tpu-vm describe tpunode16 --project raden-tpu --zone=europe-west4-a --format="value(networkEndpoints[0].accessConfig.externalIp)"
34.34.119.30

ssh -o ControlMaster=auto -o ControlPersist=yes -o ControlPath=~/.ssh/controlmasters/tpu8-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.34.119.30
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split train --out_dir /dev/shm/imagenet64
uv run python image_lagcodec/scripts/imagenet/download_imagenet64.py --split validation --out_dir /dev/shm/imagenet64

rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" . muaz@34.34.119.30:~/qcute/


rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu16-%r@%h:%p -i ~/.ssh/google_compute_engine" --filter=':- ../.gitignore' --exclude=".git/" image_lagcodec/ muaz@34.34.119.30:~/qcute/image_lagcodec/

rsync -avz --exclude="checkpoints/" -e "ssh -o ControlPath=~/.ssh/controlmasters/tpu34-%r@%h:%p -i ~/.ssh/google_compute_engine" muaz@34.34.119.30:~/qcute/image_lagcodec/logs/imagenet64_par1 image_lagcodec/logs/
