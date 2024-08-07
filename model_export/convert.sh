mkdir models
b=1
pushd tmp

## vae decoder
pushd vae
name=vae_decoder
shape=[[1,16,128,128]]
model_transform.py --model_name $name --input_shape $shape --model_def $name''.onnx --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize F16 --chip bm1684x --model $name.bmodel
rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$' | grep -v '.pt$' | grep -v 'profile_0.txt$'`
cp *.bmodel ../../models
popd
# ##

## mmdit
pushd ./mmdit
rm -rf `ls | grep '.bmodel$' | grep -v mmdit.bmodel`
rm -rf `ls | grep '.mlir$'`
dtype=BF16
name=head
# quant="--quant_input  --quant_output"
quant=""
shape=[[1,16,128,128],[1,154,4096],[1,2048],[1]]
model_transform.py --model_name mmdit_$name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $dtype --chip bm1684x $quant --model $name.bmodel
rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$' | grep -v '.pt$' | grep -v 'profile_0.txt$'`

block_num=23

shape=[[1,4096,1536],[1,1536],[1,154,1536]]
for i in $(seq 0 $block_num);
do
    name=block_$i
    model_transform.py --model_name mmdit_$name --input_shape $shape --model_def $name.pt --mlir $name.mlir 
    model_deploy.py --mlir $name.mlir --quantize $dtype --chip bm1684x  $quant --model $name.bmodel
    rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$' | grep -v '.pt$' | grep -v 'profile_0.txt$'`
done

name=tail
shape=[[1,4096,1536],[1,1536]]
model_transform.py --model_name mmdit_$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize F32  $quant --chip bm1684x --model $name.bmodel
rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$' | grep -v '.pt$' | grep -v 'profile_0.txt$'`

model_tool --combine `ls | grep '.bmodel$' | grep -v mmdit.bmodel`  -o mmdit.bmodel

cp *.bmodel ../../models
popd
#

# exit 0
# pushd ./mmdit
# --test_input ../test_block.npz --test_result block_top_res.npz
# block_num=23
# shape=[[1,4096,1536],[1,1536],[1,154,1536]]



# ## t5
pushd ./t5
name=head
shape=[[1,77]]
# quant="--quant_input  --quant_output"
quant=""
dtype=BF16
model_transform.py --model_name t5_encoder_$name --input_shape $shape --model_def t5_encoder_$name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize F16 --chip bm1684x $quant --model $name.bmodel
rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.pt$' | grep -v 'profile_0.txt$'` 
num=23
for i in $(seq 0 $num);
do
    name=block$i
    shape=[[1,77,4096]]
    # model_transform.py --model_name t5_encoder_$name --input_shape $shape --model_def t5_encoder_$name.onnx --mlir $name.mlir
    model_transform.py --model_name t5_encoder_$name --input_shape $shape --model_def t5_encoder_$name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize W4BF16 --chip bm1684x $quant --model $name.bmodel
    rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$' | grep -v '.pt$'  | grep -v 'profile_0.txt$' `
done
name=tail
shape=[[1,77,4096]]
model_transform.py --model_name t5_encoder_$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize F16 --chip bm1684x $quant --model $name.bmodel
rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$' | grep -v '.pt$' | grep -v 'profile_0.txt$'` 
model_tool --combine `ls | grep '.bmodel$' | grep -v t5.bmodel`  -o t5.bmodel
cp t5.bmodel ../../models
popd
# ##

# clip_l
pushd ./clip_l
name=head
shape=[[1,77]]
model_transform.py --model_name clip_l_$name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize F16 --chip bm1684x --quant_input  --quant_output --model $name.bmodel
rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$' | grep -v 'profile_0.txt$'` 
num=31
for i in $(seq 0 $num);
do
    name=block_$i
    shape=[[1,77,1280]]
    model_transform.py --model_name clip_l_$name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize F16 --chip bm1684x --quant_input  --quant_output --model $name.bmodel
    rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$'  | grep -v 'profile_0.txt$' `
done
name=tail
shape=[[1,77,1280],[1,77]]
model_transform.py --model_name clip_l_$name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize F16 --chip bm1684x --quant_input  --quant_output --model $name.bmodel
rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$' | grep -v 'profile_0.txt$'` 
model_tool --combine `ls | grep '.bmodel$' | grep -v clip_l.bmodel`  -o clip_l.bmodel
cp *.bmodel ../../models
popd
## 

# ## clip_g
pushd ./clip_g
name=head
shape=[[1,77]]
model_transform.py --model_name clip_g_$name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize F16 --chip bm1684x --quant_input  --quant_output --model $name.bmodel
rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$' | grep -v 'profile_0.txt$'`
num=11
for i in $(seq 0 $num);
do
    name=block_$i
    shape=[[1,77,768]]
    model_transform.py --model_name clip_g_$name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize F16 --chip bm1684x --quant_input  --quant_output --model $name.bmodel
    rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$'  | grep -v 'profile_0.txt$' `
done
name=tail
shape=[[1,77,768],[1,77]]
model_transform.py --model_name clip_g_$name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize F16 --chip bm1684x --quant_input  --quant_output --model $name.bmodel
rm -rf `ls | grep -v '.bmodel$' | grep -v '.mlir$' | grep -v '.onnx$' | grep -v 'profile_0.txt$'`
model_tool --combine `ls | grep '.bmodel$' | grep -v clip_g.bmodel`  -o clip_g.bmodel
cp clip_g.bmodel ../../models
popd
# ##