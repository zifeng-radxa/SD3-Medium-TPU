import ctypes
import numpy as np
import torch
import os
import platform
from typing import Any
import pdb

randn_tensor = lambda shape, dtype: torch.randn(shape, dtype=dtype)
SD3_DEBUG=0

def debug_print(*args,**kwargs):
    if SD3_DEBUG > 0:
        print("debug: ", end="")
        print(*args, **kwargs)
        print(flush=True)

def make_np2c(np_array:np.ndarray):
    if np_array.flags['CONTIGUOUS'] == False:
        # info users
        np_array = np.ascontiguousarray(np_array)
    return np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

def make_torch2c(tensor:torch.Tensor):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    ptr = tensor.data_ptr()
    return ctypes.c_void_p(ptr)

int_point = ctypes.POINTER(ctypes.c_int)
int_      = ctypes.c_int
ulonglong = ctypes.c_ulonglong
cpoint    = ctypes.c_void_p
vpoint    = ctypes.c_void_p
spoint    = ctypes.c_char_p
bool_     = ctypes.c_bool
null_ptr  = ctypes.c_void_p(None)
ref       = lambda x: ctypes.byref(x)

def make2_c_uint64_list(my_list):
    return (ctypes.c_uint64 * len(my_list))(*my_list)

def make2_c_int_list(my_list:list):
    return (ctypes.c_int * len(my_list))(*my_list)

def char_point_2_str(char_point:ctypes.c_char_p):
    return ctypes.string_at(char_point).decode('utf-8')

def str2char_point(string:str):
    return ctypes.c_char_p(string.encode('utf-8'))

def make2_c_point_list(my_list:list):
    return (ctypes.c_void_p * len(my_list))(*my_list)
    
def build_c_torch_lists(args):
    # need be torch
    return make2_c_point_list( [ make_torch2c(i) for i in args] )

str2cpoint = str2char_point

class Builder:

    def __init__(self, so_path:str="./build/libsd3.so"):
        self.so_path = so_path
        self.lib = ctypes.CDLL(self.so_path)
        self.lib_init()

    def lib_init(self):
        # t5 model 
        # struct t5_encoder * t5_encoder_init(const char* filename, int device_id, const char* weight_file);
        self.lib.t5_encoder_init.argtypes = [spoint, int_, spoint]
        self.lib.t5_encoder_init.restype  = vpoint
        # int t5_encoder_run(struct t5_encoder *encoder, void* data, void* output);
        self.lib.t5_encoder_run.argtypes = [vpoint, cpoint, cpoint]
        self.lib.t5_encoder_run.restype  = int_
        # int t5_encoder_free(struct t5_encoder *encoder);
        self.lib.t5_encoder_free.argtypes = [vpoint]
        self.lib.t5_encoder_free.restype  = int_

        # struct mmdit * mmdit_init(const char* filename, int device_id);
        self.lib.mmdit_init.argtypes = [spoint, int_]
        self.lib.mmdit_init.restype  = vpoint
        # int mmdit_run(struct mmdit *mmdit, void* data0, void* data1, void* data2, void* output);
        self.lib.mmdit_run.argtypes = [vpoint, cpoint, cpoint, cpoint, cpoint]
        self.lib.mmdit_run.restype  = int_
        # int mmdit_free(struct mmdit *mmdit);
        self.lib.mmdit_free.argtypes = [vpoint]
        self.lib.mmdit_free.restype  = int_

        # struct clip_l_encoder * clip_l_encoder_init(const char* filename, int device_id);
        # int clip_l_encoder_run(struct clip_l_encoder *encoder, void* input_tokens, int clip_skip, void* prompt_embed, void* pooling_embed)
        # int clip_l_free(struct clip_l_encoder *encoder);
        self.lib.clip_l_encoder_init.argtypes = [spoint, int_]
        self.lib.clip_l_encoder_init.restype  = vpoint
        self.lib.clip_l_encoder_run.argtypes  = [vpoint, cpoint, int_, cpoint, cpoint]
        self.lib.clip_l_encoder_run.restype   = int_
        self.lib.clip_l_free.argtypes = [vpoint]
        self.lib.clip_l_free.restype  = int_

        # struct clip_g_encoder * clip_g_encoder_init(const char* filename, int device_id);
        # int clip_g_encoder_run(struct clip_g_encoder *encoder, void* input_tokens, int clip_skip, void* prompt_embed, void* pooling_embed);
        # int clip_g_free(struct clip_g_encoder *encoder);
        self.lib.clip_g_encoder_init.argtypes = [spoint, int_]
        self.lib.clip_g_encoder_init.restype  = vpoint
        self.lib.clip_g_encoder_run.argtypes  = [vpoint, cpoint, int_, cpoint, cpoint]
        self.lib.clip_g_encoder_run.restype   = int_
        self.lib.clip_g_free.argtypes = [vpoint]
        self.lib.clip_g_free.restype  = int_

        # struct vae_decoder* vae_decoder_init(const char* filename, int device_id);
        # int vae_decoder_run(struct vae_decoder *decoder, void* latent, void* img, bool do_post_process);
        # int vae_decoder_free(struct vae_decoder *decoder);
        self.lib.vae_decoder_init.argtypes = [spoint, int_]
        self.lib.vae_decoder_init.restype  = vpoint
        self.lib.vae_decoder_run.argtypes  = [vpoint, cpoint, cpoint, bool_]
        self.lib.vae_decoder_run.restype   = int_
        self.lib.vae_decoder_free.argtypes = [vpoint]
        self.lib.vae_decoder_free.restype  = int_

        # void reorder_inverse(void* input, void* output, int n, int h, int w, int p, int q, int c, int dtype_len)
        self.lib.reorder_inverse.argtypes = [cpoint, cpoint, int_, int_, int_, int_, int_, int_, int_]

        # void run_any_part_model(void* model_struct, const char* model_name, const char* part_name, void** inputs, int input_num, void** outputs, int output_num);
        #self.lib.run_any_part_model.argtypes = [vpoint, spoint, spoint, cpoint, int_, cpoint, int_]

        # void run_model(const char* model_path, const char* part_name, void** inputs, int input_num, void** outputs, int output_num);
        #self.lib.run_model.argtypes = [spoint, spoint, cpoint, int_, cpoint, int_]



class T5Bmodel:
    def __init__(self, path, device_id, cpu_weight_path, batch=1, max_seq = 77, hidden_size=4096, return_dytpe=torch.float32, builder=None):
        self.builder = builder
        self.model   = builder.lib.t5_encoder_init( str2char_point(path), device_id, str2char_point(cpu_weight_path))
        self.batch   = batch
        self.max_seq = max_seq
        self.hidden_size = hidden_size
        self.dtype = return_dytpe
        self.path  = path

    def __call__(self, input_tokens, *args, **kwargs):
        input_tokens = input_tokens.int()
        res = torch.zeros( self.batch, self.max_seq, self.hidden_size, dtype=self.dtype)
        self.builder.lib.t5_encoder_run(self.model, 
                                        make_torch2c(input_tokens), 
                                        make_torch2c(res))
        return res

    def __del__(self):
        self.builder.lib.t5_encoder_free(self.model)

class MMDiTBmodel:
    def __init__(self, path, device_id, batch=1, channel=16, maxh=128, maxw=128, tokens=77*2, pooled_project_dim=2048, return_dtype=torch.float16, builder=None):
        self.builder = builder
        self.model = builder.lib.mmdit_init( str2char_point(path), device_id)
        self.batch = batch
        self.channel = channel
        self.maxh = maxh
        self.maxw = maxw
        self.tokens = tokens
        self.pool_dim = pooled_project_dim
        self.dtype = return_dtype
        self.path = path

    def forward_one_batch(self,init_states, init_encoder_hidden_states, pooled_projects, timestep):
        init_states                = init_states.to(self.dtype)
        init_encoder_hidden_states = init_encoder_hidden_states.to(self.dtype)
        pooled_projects            = pooled_projects.to(self.dtype)
        # todo check shape and dtype
        timestep = timestep.float()
        res = torch.zeros( self.batch, self.channel, self.maxh, self.maxw, dtype=self.dtype  )
        self.builder.lib.mmdit_run(self.model, 
                                   make_torch2c(init_states), 
                                   make_torch2c(init_encoder_hidden_states),
                                   make_torch2c(pooled_projects),
                                   make_torch2c(timestep),
                                   make_torch2c(res))
        return res

    def __del__(self):
        self.builder.lib.mmdit_free(self.model)
    
    def __call__(self, init_states, init_encoder_hidden_states, pooled_projects, timestep, *args, **kwargs):
        res = self.forward_one_batch(init_states, init_encoder_hidden_states, pooled_projects, timestep)
        return res

class Clip_L_Encoder:
    def __init__(self, path, device_id, batch=1, max_seq=77, hidden_size=1280, return_dytpe=torch.float16, builder=None):
        self.builder = builder
        self.model = builder.lib.clip_l_encoder_init( str2char_point(path), device_id)
        self.batch = batch
        self.max_seq = max_seq
        self.hidden_size = hidden_size
        self.dtype = return_dytpe

    def __call__(self, input_tokens, clip_skip, *args, **kwargs):
        input_tokens = input_tokens.int()
        prompt_embeds = torch.zeros(self.batch, self.max_seq, self.hidden_size, dtype=self.dtype)
        pooling_embeds = torch.zeros(self.batch, self.hidden_size, dtype=self.dtype)
        debug_print("clip_l_encoder_run")
        debug_print(input_tokens.shape, input_tokens.dtype, clip_skip, prompt_embeds.shape, pooling_embeds.shape)
        self.builder.lib.clip_l_encoder_run(self.model, 
                                            make_torch2c(input_tokens), 
                                            clip_skip,
                                            make_torch2c(prompt_embeds),
                                            make_torch2c(pooling_embeds))
        return prompt_embeds, pooling_embeds

    def __del__(self):
        self.builder.lib.clip_l_free(self.model)

class Clip_G_Encoder:
    def __init__(self, path, device_id, batch=1, max_seq=77, hidden_size=768, return_dytpe=torch.float16, builder=None):
        self.builder = builder
        self.model = builder.lib.clip_g_encoder_init( str2char_point(path), device_id)
        self.batch = batch
        self.max_seq = max_seq
        self.hidden_size = hidden_size
        self.dtype = return_dytpe

    def __call__(self, input_tokens, clip_skip, *args, **kwargs):
        input_tokens = input_tokens.int()
        prompt_embeds = torch.zeros(self.batch, self.max_seq, self.hidden_size, dtype=self.dtype)
        pooling_embeds = torch.zeros(self.batch, self.hidden_size, dtype=self.dtype)
        debug_print("clip_g_encoder_run")
        debug_print(input_tokens.shape, input_tokens.dtype, clip_skip, prompt_embeds.shape, pooling_embeds.shape)
        self.builder.lib.clip_g_encoder_run(self.model, 
                                            make_torch2c(input_tokens), 
                                            clip_skip,
                                            make_torch2c(prompt_embeds),
                                            make_torch2c(pooling_embeds))
        return prompt_embeds, pooling_embeds
    
    def __del__(self):
        self.builder.lib.clip_g_free(self.model)

class Vae_Decoder:
    def __init__(self, path, device_id, batch=1, h=128, w=128, c=16, oc=3, oh=1024, ow=1024, return_dtype=torch.float16, builder=None):
        self.builder = builder
        self.model = builder.lib.vae_decoder_init( str2char_point(path), device_id)
        self.batch = batch
        self.h = h
        self.w = w
        self.c = c
        self.oc = oc
        self.oh = oh
        self.ow = ow
        self.dtype = return_dtype
    
    def __call__(self, latent, *args, **kwargs):
        latent = latent.to(self.dtype)
        img = torch.zeros(self.batch, self.oc, self.oh, self.ow, dtype=self.dtype)
        self.builder.lib.vae_decoder_run(self.model, 
                                         make_torch2c(latent), 
                                         make_torch2c(img),
                                         False)
        return img

    def __del__(self):
        self.builder.lib.vae_decoder_free(self.model)

from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, T5TokenizerFast
from tqdm.auto import tqdm 
from PIL import Image


class SD3Pipeline:

    def __init__(self, mmdit, 
                 text_encoder1, 
                 text_encoder2, 
                 text_encoder3, 
                 vae,
                 tokenizer_path, 
                 tokenizer2_path, 
                 tokenizer3_path,
                 t5_cpu_weight,
                 builder=None
                 ):
        self.mmdit          = MMDiTBmodel(mmdit, 0, return_dtype=torch.float32, builder=builder)
        self.text_encoder   = Clip_G_Encoder(text_encoder1, 0, builder=builder)
        self.text_encoder_2 = Clip_L_Encoder(text_encoder2, 0, builder=builder)
        self.text_encoder_3 = T5Bmodel(text_encoder3, 0, t5_cpu_weight, builder=builder)
        self.vae            = Vae_Decoder(vae, 0, return_dtype=torch.float32, builder=builder)
        self.scheduler      = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        self.tokenizer      = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer_2    = CLIPTokenizer.from_pretrained(tokenizer2_path)
        self.tokenizer_3    = T5TokenizerFast.from_pretrained(tokenizer3_path)
        # pdb.set_trace()
    
    def encoder_once_prompt(self, prompt, prompt_2, prompt_3, clip_skip=0):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        prompt_2 = prompt_2 if prompt_2 else prompt
        prompt_3 = prompt_3 if prompt_3 else prompt
        debug_print("token output")
        debug_print(text_input_ids)
        prompt_embeds, pooling_embeds = self.text_encoder(text_input_ids, clip_skip)
        debug_print(prompt_embeds, pooling_embeds)
        text_inputs_2 = self.tokenizer_2(prompt_2,
                                        padding="max_length",
                                        max_length=77,
                                        truncation=True,
                                        return_tensors="pt"
                                       ).input_ids
        debug_print("token 2 output")
        debug_print(text_inputs_2)
        prompt_embeds_2, pooling_embeds_2 = self.text_encoder_2(text_inputs_2, clip_skip)
        debug_print(prompt_embeds_2, pooling_embeds_2)
        text_inputs_3 = self.tokenizer_3(prompt_3,
                                        padding="max_length",
                                        max_length=77,
                                        truncation=True,
                                        add_special_tokens=True,
                                        return_tensors="pt"
                                       ).input_ids
        t5_prompt_embed = self.text_encoder_3(text_inputs_3)
        clip_prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        clip_prompt_embeds = torch.nn.functional.pad(
                        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
                    )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooling_embeds, pooling_embeds_2], dim=-1)


        return prompt_embeds, pooled_prompt_embeds
    
    def denoise_loop(self, latent_model_input, timestep):
        # later we will move batch into c to get higher performance 
        res_no_cond, res_cond = None, None
        if self.do_cfg:
            res_no_cond = self.mmdit(latent_model_input, self.neg_encoder_hidden, self.neg_pooled_projections, timestep)
            # any nan or inf
            if torch.isnan(res_no_cond).any() or torch.isinf(res_no_cond).any():
                print("nan or inf in res_no_cond")
                pdb.set_trace()
        res_cond = self.mmdit(latent_model_input, self.pos_encoder_hidden, self.pos_pooled_projections, timestep)
        if torch.isnan(res_cond).any() or torch.isinf(res_cond).any():
            print("nan or inf in res_cond")
            pdb.set_trace()
        return res_no_cond, res_cond
    
    def __call__(self, 
                prompt=None,
                prompt_2=None,
                prompt_3=None,
                negative_prompt=None,
                negative_prompt_2=None,
                negative_prompt_3=None,
                height=1024,
                width=1024,
                clip_skip=0,
                num_inference_steps=20,
                guidance_scale=7.0,
                generator=None,
                timesteps = None
                ):
        device = "cpu"
        prompt_2 = prompt_2 if prompt_2 else prompt
        prompt_3 = prompt_3 if prompt_3 else prompt 
        negative_prompt_2 = negative_prompt_2 if negative_prompt_2 else negative_prompt
        negative_prompt_3 = negative_prompt_3 if negative_prompt_3 else negative_prompt
        self.guidance_scale = guidance_scale
        do_cfg = guidance_scale > 1
        self.do_cfg = do_cfg
        
        pos_encoder_hidden, pos_pooled_projections = self.encoder_once_prompt(prompt, prompt_2, prompt_3, clip_skip)
        neg_encoder_hidden, neg_pooled_projections = None, None
        if self.do_cfg:
            neg_encoder_hidden, neg_pooled_projections = self.encoder_once_prompt(negative_prompt, negative_prompt_2, negative_prompt_3, clip_skip)
        self.pos_encoder_hidden     = pos_encoder_hidden
        self.pos_pooled_projections = pos_pooled_projections
        self.neg_encoder_hidden     = neg_encoder_hidden
        self.neg_pooled_projections = neg_pooled_projections

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        latents = randn_tensor((1, 16, 1024//8, 1024//8), torch.float16)
        
        for t in tqdm(timesteps):
            latent_model_input = latents
            timestep = t
            noise_pred_uncond, noise_pred  = self.denoise_loop(latent_model_input, timestep)
            if self.do_cfg:
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        img = self.vae(latents / 1.5305 + 0.0609)
        # all can be done in tpu
        img = (img / 2 + 0.5).clamp(0,1).numpy()
        img = (img*255).round().astype("uint8").transpose(0,2,3,1)
        return Image.fromarray(img[0])


def test_t5():
    builder  = Builder("./build/libsd3.so")
    t5 = T5Bmodel("./models/t5.bmodel", 2, "./t5_encoder_finnal_rms_weight.bin", builder=builder)
    t5_encoder = [[   71,  1712,  3609,     3,     9,  1320,    24,   845, 21820,   296,
                1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0]]
    t5_encoder_inputs = torch.tensor(t5_encoder,dtype=torch.int32)
    res = t5(t5_encoder_inputs)
    print(res.shape)
    print(res)
    import pdb;pdb.set_trace()

# if __name__ == "__main__":
#     test_t5()

if __name__ == "__main__":
    vae_path           = "./models/vae_decoder.bmodel"
    mmdit_path         = "./models/mmdit.bmodel"
    text_encoder_path  = "./models/clip_g.bmodel"
    text_encoder2_path = "./models/clip_l.bmodel"
    text_encoder3_path = "./models/t5.bmodel"
    t5_cpu_weight      = "./models/t5_encoder_finnal_rms_weight.bin"
    tokenizer_path     = "../token/tokenizer"
    tokenizer2_path    = "../token/tokenizer_2"
    tokenizer3_path    = "../token/tokenizer_3"
    builder  = Builder("./libsd3.so")
    pipeline = SD3Pipeline(mmdit_path, text_encoder_path, text_encoder2_path, text_encoder3_path, vae_path, tokenizer_path, tokenizer2_path, tokenizer3_path, 
                           t5_cpu_weight,
                           builder=builder)

    res = pipeline("A cat with a sign text Welcome to radxa!", negative_prompt="deformed, lowres, bad anatomy, error, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, artist name", num_inference_steps=28)
    res.save("result.png")