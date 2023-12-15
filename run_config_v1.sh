num=$1
device=$2
echo $num
echo $device
# docker run --cap-add SYS_ADMIN -itd --restart always -p 1100$num:7019 --device /dev/bmdev-ctl --device=/dev/bm-sophon$device:/dev/bm-sophon0 --log-opt max-size=16g --log-opt max-file=1 -v /data/user/aigc/bb/aigc/models:/workspace/models -w /workspace --name aigc$num aigc bash
# git checkout test
            # -v /home/zhongying.ru/aaaigc/aigc/models:/workspace/models \
# docker run --cap-add SYS_ADMIN -itd -p 1100$num:7019 --device /dev/bmdev-ctl --device=/dev/bm-sophon$device:/dev/bm-sophon0 --log-opt max-size=16g --log-opt max-file=1 -v /home/zhongying.ru/code/aaaigc/aigc/models:/workspace/models -v /home/zhongying.ru/code/aaaigc/aigc:/workspace -w /workspace --name aigc$num aigc bash run.sh

# for test --restart always \
docker run  --cap-add SYS_ADMIN \
            -itd \
            -p 1110$num:7019 \
            --restart always \
            --device /dev/bmdev-ctl \
            --device=/dev/bm-sophon$device:/dev/bm-sophon0 \
            --log-opt max-size=16g \
            --log-opt max-file=1 \
            -v /data/aigc/aigc/models:/workspace/models/ \
            -v /home/zhongying.ru/aaaigc/aigc/models/controlnet:/workspace/models/controlnet \
            -v /data/aigc/aigc:/workspace \
            -w /workspace \
            --name aigc$num \
            aigc \
            bash run_v1.sh