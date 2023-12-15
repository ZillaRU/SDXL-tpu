model_transform.py  --model_name hed_process  --input_shape [1,3,512,512]  --model_def hed_processor.pt  --mlir hed_processor.mlir
model_deploy.py  --mlir hed_processor.mlir  --quantize F32  --chip bm1684x  --model hed_processor_1684x_f32.bmodel
