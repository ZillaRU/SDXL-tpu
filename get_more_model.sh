echo "Please check 'models' folder for more models."
echo "the model path is in : url"

mkdir models/basic/babes20
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/SmJpQmYpT && mv unet_2_1684x_f16.bmodel models/basic/babes20/
echo "reuse the encoder and vae model from deliberate-lora_pixarStyleLora_lora128-unet-2"
cp models/basic/deliberate-lora_pixarStyleLora_lora128-unet-2/*vae*model models/basic/babes20/
cp models/basic/deliberate-lora_pixarStyleLora_lora128-unet-2/encoder*bmodel models/basic/babes20/
echo "babes20 model is ready"
echo "if you want to use babes20 model, please run export BASENAME='babes20' before run.sh"