export PATH=$PATH:/opt/sophon/libsophon-current/bin
export PATH=$PATH:/opt/sophon/libsophon-current/bin/bm1684x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/libsophon-current/lib
# export BASENAME=`ls models/basic`
export BASENAME='babes20' 
#`ls models/basic` # 'babs_20_nolora' #'flat2DAnimerge_v10-unet-2'
# export CONTROLNET=`ls models/controlnet/ | grep -v tile | awk -F "." '{print $1}'`

echo $BASENAME
python3 -m pip install lark
python3 app.py