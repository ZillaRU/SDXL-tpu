# # 如果 babses20 在 models/basic 下，那么export BASENAME='babes20'，否则export BASENAME='basic/deliberate-lora_pixarStyleLora_lora128-unet-2'
# export DEVICE_ID=8
# # export BASENAME='delibrate'
# export BASENAME='SDXL'
# # if [ -d "./models/basic/babes20" ]; then
# #      export BASENAME='model9'
# # fi
# # if [ $(uname -m) = "x86_64" ]; then
# #      echo "Your current operating system is based on x86_64"
# # else
# #      export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
# # fi

# python3 -m pip install lark
# python3 app_xl.py 

export PATH=$PATH:/opt/sophon/libsophon-current/bin
export PATH=$PATH:/opt/sophon/libsophon-current/bin/bm1684x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/libsophon-current/lib
# export BASENAME=`ls models/basic`
export BASENAME='SDXL' 
#`ls models/basic` # 'babs_20_nolora' #'flat2DAnimerge_v10-unet-2'
# export CONTROLNET=`ls models/controlnet/ | grep -v tile | awk -F "." '{print $1}'`

echo $BASENAME
# echo $CONTROLNET
apt update && apt install -y dkms libncurses5 && dpkg -i ./debs/sophon-*.deb
source /etc/profile
# bmrt_test --bmodel /workspace/models/basic/antalpha-sd2-1-ep20-gs529400/unet_multize.bmodel
python3 -m pip install lark
# python3 app.py
python3 app_xl.py 