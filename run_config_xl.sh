num=$1
device=$2
echo $num
echo $device

# for test --restart always \
docker run  --cap-add SYS_ADMIN \
            -itd \
            -p 1110$num:7019 \
            --device /dev/bmdev-ctl \
            --device=/dev/bm-sophon$device:/dev/bm-sophon0 \
            --log-opt max-size=16g \
            --log-opt max-file=1 \
            -v /data/aigc/aigc/models:/workspace/models/ \
            -v /home/zhongying.ru/aaaigc/aigc/models/controlnet:/workspace/models/controlnet \
            -v /data/aigc/aa/sdxl/models:/workspace/xl_models \
            -v /data/aigc/aigc:/workspace \
            -w /workspace \
            --name aigc$num \
            aigc \
            bash run_xl.sh
