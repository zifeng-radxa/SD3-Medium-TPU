import gradio as gr
from sd3bmodel import SD3Pipeline, Builder
import os
import warnings
from utils.tools import create_size, get_model_path, seed_torch

warnings.filterwarnings("ignore")
model_path = get_model_path()

DEVICE_ID = 0

SIZE = create_size(1024)  # [('512:512', [512,512]), ] W, H


class ModelManager():
    def __init__(self):
        self.current_model_name = None
        self.pipe = None

    def pre_check(self, model_select="sd3", check_type=None):
        sd3_model_path = []
        check_pass = True
        model_select_path = os.path.join('models')
        clip_g_path = os.path.join(model_select_path, model_path[model_select]['clip_g'])
        clip_l_path = os.path.join(model_select_path, model_path[model_select]['clip_l'])
        mmdit_path = os.path.join(model_select_path, model_path[model_select]['mmdit'])
        t5_path = os.path.join(model_select_path, model_path[model_select]['t5'])
        vae_de_path = os.path.join(model_select_path, model_path[model_select]['vae_decoder'])

        if "clip_g" in check_type:
            if not os.path.isfile(clip_g_path):
                gr.Warning("No {} clip_g, please download first".format(model_select))
                check_pass = False
                # return False
        if "clip_l" in check_type:
            if not os.path.isfile(clip_l_path):
                gr.Warning("No {} clip_l, please download first".format(model_select))
                check_pass = False

        if "mmdit" in check_type:
            if not os.path.exists(mmdit_path):
                gr.Warning("No {} mmdit, please download first".format(model_select))
                check_pass = False
        if "t5" in check_type:
            if not os.path.exists(t5_path):
                gr.Warning("No {} t5, please download first".format(model_select))
                check_pass = False

        if "vae" in check_type:
            if not os.path.exists(vae_de_path):
                gr.Warning("No {} vae, please download first".format(model_select))
                check_pass = False

        sd3_model_path.append(mmdit_path)
        sd3_model_path.append(clip_g_path)
        sd3_model_path.append(clip_l_path)
        sd3_model_path.append(t5_path)
        sd3_model_path.append(vae_de_path)

        return check_pass, sd3_model_path

    def change_model(self, model_select, progress=gr.Progress()):
        if model_select == []:
            model_select = None
        if model_select is not None:
            if self.pipe is None:
                check_pass, sd3_model_select = self.pre_check(check_type=["clip_g", "clip_l", "mmdit", "t5", "vae"])
                if check_pass:
                    self.pipe = SD3Pipeline(sd3_model_select[0],
                                            sd3_model_select[1],
                                            sd3_model_select[2],
                                            sd3_model_select[3],
                                            sd3_model_select[4],
                                            tokenizer_path="../token/tokenizer",
                                            tokenizer2_path="../token/tokenizer_2",
                                            tokenizer3_path="../token/tokenizer_3",
                                            t5_cpu_weight="./models/t5_encoder_finnal_rms_weight.bin",
                                            builder=Builder("./libsd3.so")
                                            )
                    self.current_model_name = model_select
                    gr.Info("{} load success".format(model_select))

                    return self.current_model_name
                else:
                    gr.Error("{} models are not complete".format(model_select))

        else:
            gr.Info("Please select a model")
            return None

    def generate_image_from_text(self,
                                 input_prompt_1,
                                 input_prompt_2,
                                 input_prompt_3,
                                 negative_prompt_1,
                                 negative_prompt_2,
                                 negative_prompt_3,
                                 latent_size=None,
                                 clip_skip=0,
                                 steps=20,
                                 guidance_scale=7.0,
                                 seed_number=0
                                 ):
        if self.pipe is None:
            gr.Info("Please select a model")
            return None

        else:
            seed_torch(seed_number)
            if input_prompt_1 == "":
                gr.Warning("please input your prompt")
                return None

            if input_prompt_2 == "":
                input_prompt_2 = None

            if input_prompt_3 == "":
                input_prompt_3 = None

            if negative_prompt_2 == "":
                negative_prompt_2 = None

            if negative_prompt_3 == "":
                negative_prompt_3 = None

            # print(input_prompt_1, type(input_prompt_1))
            # print(input_prompt_2, type(input_prompt_2))
            # print(input_prompt_3, type(input_prompt_3))
            # print(negative_prompt_1, type(negative_prompt_1))
            # print(negative_prompt_2, type(negative_prompt_2))
            # print(negative_prompt_3, type(negative_prompt_3))

            img_pil = self.pipe(input_prompt_1,
                                input_prompt_2,
                                input_prompt_3,
                                negative_prompt_1,
                                negative_prompt_2,
                                negative_prompt_3,
                                SIZE[latent_size][1][0],
                                SIZE[latent_size][1][1],
                                clip_skip,
                                steps,
                                guidance_scale,
            )

            return img_pil



model_manager = ModelManager()

description = """
# Stable Diffusion 3 Medium on Airbox 🥳
"""

if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_prompt_1 = gr.Textbox(lines=1, label="Prompt", value=None)

                with gr.Row():
                    negative_prompt_1 = gr.Textbox(lines=1, label="Negative Prompt", value=None)


                with gr.Row():
                    num_step = gr.Slider(minimum=18, maximum=40, value=20, step=1, label="Steps", scale=2)
                    guidance_scale = gr.Slider(minimum=0, maximum=20, value=0, step=0.1, label="CFG scale", scale=2)

                with gr.Row():
                    seed_number = gr.Number(value=1, label="Seed", scale=1, minimum=0)
                    clip_skip = gr.Number(value=0, step=1, label="clip skip", scale=1, minimum=0, maximum=2)
                    latent_size_index = gr.Dropdown(choices=[i[0] for i in SIZE], label="Size (W:H)",
                                                    value=[i[0] for i in SIZE][0], type="index", interactive=True,
                                                    scale=1)
                    # scheduler_type = gr.Dropdown(choices=scheduler, value=scheduler[0], label="Scheduler", interactive=True,scale=1)
                with gr.Row():
                    with gr.Accordion("Advanced", open=False):
                        input_prompt_2 = gr.Textbox(lines=1, label="CLIP_L Prompt", value=None)
                        negative_prompt_2 = gr.Textbox(lines=1, label="CLIP_L Negative Prompt", value=None)

                        input_prompt_3 = gr.Textbox(lines=1, label="T5 Prompt", value=None)
                        negative_prompt_3 = gr.Textbox(lines=1, label="T5 Negative Prompt", value=None)


                with gr.Row():
                    clear_bt = gr.ClearButton(value="Clear",
                                              components=[input_prompt_1,
                                                          input_prompt_2,
                                                          input_prompt_3,
                                                          negative_prompt_1,
                                                          negative_prompt_2,
                                                          negative_prompt_3,
                                                          num_step,
                                                          guidance_scale,
                                                          ]
                                              )
                    submit_bt = gr.Button(value="Submit", variant="primary")

            with gr.Column():
                with gr.Row():
                    model_select = gr.Dropdown(choices=["sd3"], value="sd3", label="Model", interactive=True)
                    load_bt = gr.Button(value="Load Model", interactive=True)
                out_img = gr.Image(label="Output", format="png")

        with gr.Row():
            with gr.Column():
                example = gr.Examples(
                    label="Example",
                    examples=[
                        ["A cat with a sign text Welcome to radxa!",
                         "deformed, lowres, bad anatomy, error, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
                         0,
                         28,
                         7.0,
                         "1024:1024",
                         3],

                        ["A vibrant street wall covered in colorful graffiti, the centerpiece spells \"Radxa Airbox\", in a storm of colors",
                         "deformed, lowres, bad anatomy, error, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
                         0,
                         25,
                         8.0,
                         "1024:1024",
                         99],

                        ["a cyberpunk hotel with the neon sign with the text Radxa Airbox",
                         "deformed, lowres, bad anatomy, error, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
                          0,
                          28,
                          7.0,
                         "1024:1024",
                         15],

                        ["Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                         "deformed, lowres, bad anatomy, error, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
                         0,
                         28,
                         5.0,
                         "1024:1024",
                         50],
                    ],
                    inputs=[input_prompt_1, negative_prompt_1, clip_skip, num_step, guidance_scale, latent_size_index,seed_number]
                )

        clear_bt.add(components=[out_img])
        load_bt.click(model_manager.change_model, [model_select], [model_select])

        input_prompt_1.submit(model_manager.generate_image_from_text,
                              [input_prompt_1,
                               input_prompt_2,
                               input_prompt_3,
                               negative_prompt_1,
                               negative_prompt_2,
                               negative_prompt_3,
                               latent_size_index,
                               clip_skip,
                               num_step,
                               guidance_scale,
                               seed_number],
                              [out_img]
                              )
        negative_prompt_1.submit(model_manager.generate_image_from_text,
                              [input_prompt_1,
                               input_prompt_2,
                               input_prompt_3,
                               negative_prompt_1,
                               negative_prompt_2,
                               negative_prompt_3,
                               latent_size_index,
                               clip_skip,
                               num_step,
                               guidance_scale,
                               seed_number],
                              [out_img]
                              )
        submit_bt.click(model_manager.generate_image_from_text,
                        [input_prompt_1,
                         input_prompt_2,
                         input_prompt_3,
                         negative_prompt_1,
                         negative_prompt_2,
                         negative_prompt_3,
                         latent_size_index,
                         clip_skip,
                         num_step,
                         guidance_scale,
                         seed_number],
                        [out_img]
                        )

    # 运行 Gradio 应用
    demo.queue(max_size=10)
    demo.launch(server_port=8999, server_name="0.0.0.0")

