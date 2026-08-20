64 spot Cloud TPU v5e chips in zone europe-west4-b
32 spot Cloud TPU v4 chips in zone us-central2-b
64 spot Cloud TPU v6e chips in zone us-east1-d
32 on-demand Cloud TPU v4 chips in zone us-central2-b
64 spot Cloud TPU v5e chips in zone us-central1-a
64 spot Cloud TPU v6e chips in zone europe-west4-a

https://docs.cloud.google.com/tpu/docs/queued-resources#delete_a_queued_resource_request

```
gcloud compute tpus queued-resources create tpu1 \
    --node-id tpu1 \
    --project raden-tpu \
    --zone europe-west4-a \
    --accelerator-type v6e-1 \
    --runtime-version v2-alpha-tpuv6e \
    --spot

gcloud compute tpus queued-resources describe tpu1 \
    --project raden-tpu \
    --zone europe-west4-a

gcloud compute tpus queued-resources list --project raden-tpu --zone europe-west4-a

gcloud compute tpus queued-resources delete tpu1 \
    --project raden-tpu \
    --zone europe-west4-a \
    --force \
    --async

gcloud compute tpus queued-resources ssh tpu1 --project raden-tpu --zone europe-west4-a
```