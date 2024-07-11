import torch
from diffusers import StableDiffusion3Pipeline
import torch.nn as nn
import os

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# make_dir("./tmp")
# make_dir("./tmp/vae")
# make_dir("./tmp/mmdit")
# make_dir("./tmp/t5")
# make_dir("./tmp/clip_l")
# make_dir("./tmp/clip_g")


SD3_HUGGINGFACE_PATH = "/data/aigc/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(SD3_HUGGINGFACE_PATH, torch_dtype=torch.float32)

sd3 = pipe.transformer
vae = pipe.vae

vae= vae.eval()
for para in vae.parameters():
    para.requires_grad = False

# ================= export vae model ==================== #
class VAE_Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.vae = vae
        pass

    def forward(self, hidden_states):
        hidden_states = (hidden_states / 1.5305) + 0.0609
        res = self.vae.decode(hidden_states)[0]
        return res

vae_decoder = VAE_Decoder()

# ================= export over ==================== #

fake_inputs = torch.randn(1, 16, 128, 128)
torch.onnx.export(vae_decoder, fake_inputs, "./tmp/vae/vae_decoder.onnx")

# ================= export sd3 model ==================== #

class FixedLayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super(FixedLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.norm.weight.data.fill_(1.0)
        self.norm.bias.data.fill_(0.0)
        self.eps = 1e-6

    def forward(self, x):
        # mean = x - x.mean(dim=-1, keepdim=True)
        # var = mean.pow(2).mean(dim=-1, keepdim=True) + self.eps
        # # var  = x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
        # res = mean / torch.sqrt(var)
        res = self.norm(x)
        return res

def replace_layer_norm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm) and child.elementwise_affine == False and child.eps == 1e-6:
            setattr(module, name, FixedLayerNorm(child.normalized_shape[0]))
        else:
            replace_layer_norm(child)

def replace_layer_norm_test_overflow(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm) and child.elementwise_affine == False and child.eps == 1e-6:
            setattr(module, name, torch.nn.Identity())
        else:
            replace_layer_norm(child)

replace_layer_norm(sd3)

def replace_silu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name,torch.nn.Identity())
        else:
            replace_silu(child)
replace_silu(sd3.transformer_blocks)
replace_silu(sd3.norm_out)
# replace_layer_norm_test_overflow(sd3.transformer_blocks)

batch = 1
h = w = 1024
fake_inputs_shape = [ [batch, 16, h//8, w//8], [batch, 154, 4096], [batch, 2048] ]
fake_inputs = []
fake_inputs.append( torch.randn(fake_inputs_shape[0]) )
fake_inputs.append( torch.randn(fake_inputs_shape[1]) )
fake_inputs.append( torch.randn(fake_inputs_shape[2])*0.5 )
fake_inputs.append(torch.tensor([20] ) )

save_onnx = True

for para in sd3.parameters():
    para.requires_grad = False


class SD3Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = sd3.pos_embed
        self.time_text_embed = sd3.time_text_embed
        self.context_embedder = sd3.context_embedder
        self.silu  = torch.nn.SiLU()
        # self.silu = torch.nn.Identity()
    def forward(self,hidden_states, encoder_hidden_states, pooled_projections,  timestep):
        hidden_states = self.pos_embed(hidden_states) 
        temb = self.silu(self.time_text_embed(timestep, pooled_projections))
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        temb = temb
        return hidden_states, temb, encoder_hidden_states

sd3head = SD3Head()
fake_dict = {"hidden_states": fake_inputs[0], 
                            "encoder_hidden_states":fake_inputs[1],
                            "pooled_projections": fake_inputs[2],
                             "timestep": fake_inputs[-1]}
hidden_states, temb, encoder_hidden_states = sd3head(**fake_dict)
# print(encoder_hidden_states)
if save_onnx: torch.onnx.export(sd3head, tuple(fake_inputs), "./tmp/mmdit/head.onnx")

class SD3Tail(torch.nn.Module):
    def __init__(self, patch_size=2):
        super().__init__()
        self.norm_out = sd3.norm_out
        self.proj_out = sd3.proj_out
        self.patch_size = patch_size
        self.output_channels = sd3.out_channels
        self.width  = h // 8
        self.height = w // 8

    def forward(self, hidden_states, temb):
        hidden = self.norm_out(hidden_states, temb)
        hidden = self.proj_out(hidden)
        return hidden

class SD3Block(torch.nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.block = sd3.transformer_blocks[idx]
        self.idx = idx 
        
    def forward(self, hidden_states, temb, encoder_hidden_states):
        encoder_hidden_states, hidden_states = self.block(hidden_states=hidden_states, 
                                                          encoder_hidden_states=encoder_hidden_states, 
                                                          temb=temb)
        if self.idx == 23:
            return hidden_states
        return encoder_hidden_states, hidden_states


from tqdm.auto import tqdm
sd3block = []
for i in tqdm(range(23)):
    sd3block.append(SD3Block(i).eval())
    # if save_onnx:
    #     torch.onnx.export(sd3block[-1], (hidden_states, temb, encoder_hidden_states), 
    #                       f"/data/aigc/demos/sd3models/mmdit/block_{i}.onnx",do_constant_folding=True)
    torch.jit.trace( sd3block[-1], (hidden_states, temb, encoder_hidden_states)).save(f"./tmp/mmdit/block_{i}.pt")
    encoder_hidden_states, hidden_states = sd3block[-1](hidden_states, temb, encoder_hidden_states)
    # print(encoder_hidden_states)

last_block = SD3Block(23).eval()
if save_onnx:
    torch.jit.trace( last_block, (hidden_states, temb, encoder_hidden_states)).save(f"./tmp/mmdit/block_23.pt")

hidden_states = last_block(hidden_states, temb, encoder_hidden_states)

class SD3Tail(torch.nn.Module):
    def __init__(self, patch_size=2):
        super().__init__()
        self.norm_out = sd3.norm_out
        self.proj_out = sd3.proj_out
        self.patch_size = patch_size
        self.output_channels = sd3.out_channels
        self.width  = h // 8
        self.height = w // 8

    def forward(self, hidden_states, temb):
        hidden = self.norm_out(hidden_states, temb)
        hidden = self.proj_out(hidden)
        return hidden
sd3tail = SD3Tail()
res = sd3tail(hidden_states, temb)
if save_onnx: 
    torch.jit.trace( sd3tail, (hidden_states, temb)).save(f"./tmp/mmdit/tail.pt")
    # torch.onnx.export(sd3tail, (hidden_states, temb), "/data/aigc/demos/sd3models/mmdit/tail.onnx")
print(res)
# ================= export over ==================== #


# ================= export t5 ==================== #
t5_encoder = [[   71,  1712,  3609,     3,     9,  1320,    24,   845, 21820,   296,
             1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0]]
t5_encoder_inputs = torch.tensor(t5_encoder,dtype=torch.int32)

t5model = pipe.text_encoder_3.eval()
for para in t5model.parameters():
    para.requires_grad = False

t5_model = t5model
for para in t5_model.parameters():
    para.requires_grad = False

class T5Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = t5_model.encoder.embed_tokens
    def forward(self, test_input):
        return self.emb(test_input)

t5head = T5Head().eval()
input1 = t5head(t5_encoder_inputs)

temp_value = t5model.encoder.block[0].layer[0](t5model.encoder.embed_tokens(t5_encoder_inputs))[2:][0].detach().requires_grad_(False)
t = torch.zeros(1,1,1,77)
class T5block0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = t5model.encoder.block[0].layer
        
    def forward(self, test_input):
        output = self.block0[0](test_input, t, temp_value)
        hidden_states = output[0]
        hidden_states = self.block0[-1](hidden_states)
        return hidden_states
t5block0 = T5block0().eval()
torch.jit.trace(t5block0, input1).save("./tmp/t5/t5_encoder_block0.pt")
hidden_states = t5block0(input1)
# print(0, hidden_states.min(), hidden_states.max())

from tqdm.auto import tqdm
class T5blocknext(torch.nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.block = t5model.encoder.block[idx].layer
    def forward(self, test_input):
        next_block = self.block
        output = next_block[0](test_input,t, temp_value)
        hidden_states = output[0]
        hidden_states = next_block[1](hidden_states)
        return hidden_states

t5_next_blocks = []
for i in tqdm(range(1,24)):
    block = T5blocknext(i).eval()
    # torch.onnx.export(block, hidden_states, f"/data/aigc/demos/sd3models/t5/t5_encoder_block{i}.onnx")
    torch.jit.trace(block, input1).save(f"./tmp/t5/t5_encoder_block{i}.pt")
    hidden_states = block(hidden_states)
    print(i, hidden_states.max(), hidden_states.min() )

torch.jit.trace(t5head, t5_encoder_inputs).save("./tmp/t5/t5_encoder_head.pt")
torch.jit.trace(t5model.encoder.final_layer_norm, hidden_states).save("./tmp/t5/tail.pt")
res = t5model.encoder.final_layer_norm(hidden_states)
# res
# ================= export over ==================== #

# ================= export clip ==================== #
max_seq_len = 77
test_input = torch.randint(0,1000,(1,max_seq_len))
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
causal_attention = _create_4d_causal_attention_mask([1,max_seq_len], torch.float32, device="cpu")
clip0_model = pipe.text_encoder_2
clip0_model = clip0_model.eval()
class CLIPHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = clip0_model.text_model.embeddings
    def forward(self, x):
        return self.emb(x)
head = CLIPHead()
torch.onnx.export(head, test_input, "./tmp/clip_l/head.onnx")
input1 = head(test_input)
class CLIPBlock(torch.nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.block = clip0_model.text_model.encoder.layers[idx]
    def forward(self, x):
        return self.block(x, None, causal_attention)[0]
for i in range(32):
    block0 = CLIPBlock(i)
    torch.onnx.export(block0, input1, f"./tmp/clip_l/block_{i}.onnx")
    input1 = block0(input1)
res_t = clip0_model.text_model.final_layer_norm(input1)
torch.onnx.export(clip0_model.text_model.final_layer_norm, input1, f"./tmp/clip_l/tail.onnx")


# ================= export over ==================== #


# ================= export clip ==================== #
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
causal_attention = _create_4d_causal_attention_mask([1,max_seq_len], torch.float32, device="cpu")
clip0_model = pipe.text_encoder
clip0_model = clip0_model.eval()
class CLIPHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = clip0_model.text_model.embeddings
    def forward(self, x):
        return self.emb(x)
head = CLIPHead()
torch.onnx.export(head, test_input, "./tmp/clip_g/head.onnx")
input1 = head(test_input)
class CLIPBlock(torch.nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.block = clip0_model.text_model.encoder.layers[idx]
    def forward(self, x):
        return self.block(x, None, causal_attention)[0]
for i in range(12):
    block0 = CLIPBlock(i)
    torch.onnx.export(block0, input1, f"./tmp/clip_g/block_{i}.onnx")
    input1 = block0(input1)
res_t = clip0_model.text_model.final_layer_norm(input1)
torch.onnx.export(clip0_model.text_model.final_layer_norm, input1, f"./tmp/clip_g/tail.onnx")
# ================= export over ==================== #