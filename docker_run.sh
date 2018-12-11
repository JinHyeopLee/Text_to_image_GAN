docker run \
	--runtime=nvidia \
	-d \
	-v /mnt/ssd0/JH/CUB:/data \
	-v /mnt/ssd0/JH/DS_SJE_tensorflow:/project \
	-v /mnt/hdd0/JH/DS_SJE_model:/result \
	-it \
	--memory=12g \
	--memory-swap=12g \
	--cpuset-cpus 0-4,10-14 \
	--name DS_SJE_test \
	-e NVIDIA_VISIBLE_DEVICES=0 \
	tensorflow/tensorflow:latest-gpu-py3 \
    python /project/main.py
