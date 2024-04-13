import os
import json
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from conclvd.models.unet import UNet3DConditionModel
from conclvd.pipelines.pipeline_animation import AnimationPipeline
from conclvd.utils.util import save_videos_grid
from conclvd.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, \
    convert_ldm_vae_checkpoint
from conclvd.utils.convert_lora_safetensor_to_diffusers import convert_lora
import cv2
import numpy as np

sample_idx = 0
scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}

css = """
body, html {
    height: 100%;
    margin: 0;
    padding: 0;
    font-family: 'Arial', sans-serif;
    background: #f5f5dc !important; /* 宣纸灰作为页面背景色 */
}

/* 对于所有的按钮和输入控件 */
button, input, select, textarea {
    background-color: #fff5e6  !important; /* 墨绿色背景 */
    color: #000 !important; /* 黑色字体，模仿墨色 */
    border: 1px solid #fff5e6  !important; /* 边框颜色 */
    padding: 10px !important; /* 内填充 */
    margin-bottom: 10px !important; /* 元素间距 */
    border-radius: 4px !important; /* 圆角边框 */
}

/* 鼠标悬停在按钮上时的效果 */
button:hover {
    background-color: #ccc !important; /* 鼠标悬停时的背景色，淡灰色 */
    color: #000 !important; /* 悬停时的字体颜色，黑色 */
}

/* 特别的控件或容器，如面板或卡片 */
.panel, .card {
    background-color: #f5f5dc !important;
    padding: 20px !important;
    border-radius: 15px !important; /* 圆角可以给人类似画卷的感觉 */
    border: 1px solid #a0a0a0 !important; /* 深色边框来定义边缘 */
    box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3) !important; /* 阴影效果以增加立体感 */
    margin-bottom: 20px !important;
    position: relative; /* 这对于伪元素是必须的 */
    overflow: hidden; /* 确保伪元素不会溢出边框 */
}

.panel::before, .card::before {
    content: '';
    position: absolute;
    top: -10px; right: -10px; bottom: -10px; left: -10px;
    background: linear-gradient(45deg, transparent 15px, #f0e8d9 0), linear-gradient(-45deg, transparent 15px, #f0e8d9 0);
    background-repeat: repeat-x, repeat-x;
    background-size: 100px 100%, 100px 100%;
    background-position: left top, right bottom;
}
.markdown-title h1 {
    text-align: center;
    font-size: 24px; /* 或更大的值，根据您的需要 */
/* ... 其他样式 ... */
"""







class AnimateController:
    def __init__(self):

        # config dirs
        self.basedir = os.getcwd()
        self.stable_diffusion_dir = os.path.join(self.basedir, "models", "StableDiffusion")
        self.motion_module_dir = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(self.basedir, "models", "DreamBooth_LoRA")
        self.savedir = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample = os.path.join(self.savedir, "sample")
        os.makedirs(self.savedir, exist_ok=True)

        self.stable_diffusion_list = []
        self.motion_module_list = []
        self.personalized_model_list = []

        self.refresh_stable_diffusion()
        self.refresh_motion_module()
        self.refresh_personalized_model()

        # config models
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.pipeline = None
        self.lora_model_state_dict = {}

        self.inference_config = OmegaConf.load("configs/inference/inference.yaml")

    def refresh_stable_diffusion(self):
        self.stable_diffusion_list = glob(os.path.join(self.stable_diffusion_dir, "*/"))

    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.personalized_model_list = [os.path.basename(p) for p in personalized_model_list]

    def update_stable_diffusion(self, stable_diffusion_dropdown):
        self.tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_dropdown, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_dropdown, subfolder="text_encoder").cuda()
        self.vae = AutoencoderKL.from_pretrained(stable_diffusion_dropdown, subfolder="vae").cuda()
        self.unet = UNet3DConditionModel.from_pretrained_2d(stable_diffusion_dropdown, subfolder="unet",
                                                            unet_additional_kwargs=OmegaConf.to_container(
                                                                self.inference_config.unet_additional_kwargs)).cuda()
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
            motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
            motion_module_state_dict = motion_module_state_dict[
                "state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
            motion_module_state_dict = {key.replace("module.", ""): value for key, value in
                                        motion_module_state_dict.items()}

            missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)

            print(len(unexpected))
            assert len(unexpected) == 0
            return gr.Dropdown.update()

    def update_base_model(self, base_model_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
            base_model_state_dict = {}
            with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)

            converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_model_state_dict, self.vae.config)
            self.vae.load_state_dict(converted_vae_checkpoint)

            converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, self.unet.config)
            self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

            self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)
            return gr.Dropdown.update()
    '''
    def update_lora_model(self, lora_model_dropdown):
        lora_model_dropdown = os.path.join(self.personalized_model_dir, lora_model_dropdown)
        self.lora_model_state_dict = {}
        if lora_model_dropdown == "none":
            pass
        else:
            with safe_open(lora_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.lora_model_state_dict[key] = f.get_tensor(key)
        return gr.Dropdown.update()
    '''
    def update_lora_model(self):
        self.lora_model_state_dict = {}
        # 既然没有 LoRA 模型可选，此处不需要进一步的逻辑

    def calculate_optical_flow(prev_frame, next_frame):
        # 将图像转换为灰度以计算光流
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # 使用Farneback方法计算光流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        return flow

    def generate_interpolated_frame(prev_frame, next_frame, flow):
        # 生成插值帧的代码
        height, width = prev_frame.shape[:2]
        # 生成新的帧位置
        new_positions = np.dstack(np.meshgrid(np.arange(width), np.arange(height))).astype(np.float32) + flow

        # 使用remap进行插值
        interpolated_frame = cv2.remap(prev_frame, new_positions, None, cv2.INTER_LINEAR)

        return interpolated_frame

    def animate(
            self,
            stable_diffusion_dropdown,
            motion_module_dropdown,
            base_model_dropdown,
            #lora_alpha_slider,
            prompt_textbox,
            #negative_prompt_textbox,
            #sampler_dropdown,
            #sample_step_slider,
            #width_slider,
            length_slider,
            #height_slider,
            #cfg_scale_slider,
            seed_textbox
    ):
        if self.unet is None:
            raise gr.Error(f"Please select a pretrained model path.")
        if motion_module_dropdown == "":
            raise gr.Error(f"Please select a motion module.")
        if base_model_dropdown == "":
            raise gr.Error(f"Please select a base DreamBooth model.")

        if is_xformers_available(): self.unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
        vae=self.vae, 
        text_encoder=self.text_encoder, 
        tokenizer=self.tokenizer, 
        unet=self.unet,
        scheduler=EulerDiscreteScheduler(**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        ).to("cuda")

        if self.lora_model_state_dict != {}:
            pipeline = convert_lora(pipeline, self.lora_model_state_dict, alpha=0.8)

        pipeline.to("cuda")

        if seed_textbox != -1 and seed_textbox != "":
            torch.manual_seed(int(seed_textbox))
        else:
            torch.seed()
        seed = torch.initial_seed()

        sample = pipeline(
            prompt_textbox,
            #negative_prompt=negative_prompt_textbox,
            num_inference_steps=25,
            guidance_scale=7.5,
            width=384,
            height=384,
            video_length=length_slider,
        ).videos  # sample的形状为torch.Size([1, 3, 16, 384, 384])
        sample_np = sample.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()

        def calculate_optical_flow(prev_frame, next_frame):  # 计算两帧之间的光流
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            return flow

        def generate_interpolated_frame(prev_frame, next_frame, flow):
            height, width = prev_frame.shape[:2]
            # 生成新位置网格，用于映射
            new_positions = np.dstack(np.meshgrid(np.arange(width), np.arange(height))).astype(np.float32) + flow
            # 使用光流将 prev_frame 映射到新位置，生成插值帧
            interpolated_frame_from_prev = cv2.remap(prev_frame, new_positions, None, cv2.INTER_LINEAR)
            # 反向光流：假设光流是双向相等的（仅为简化示例，实际应用中可能需要单独计算）
            inverse_flow = -flow
            # 使用反向光流将 next_frame 映射到与 prev_frame 相同的时间点
            new_positions_inverse = np.dstack(np.meshgrid(np.arange(width), np.arange(height))).astype(
                np.float32) + inverse_flow
            interpolated_frame_from_next = cv2.remap(next_frame, new_positions_inverse, None, cv2.INTER_LINEAR)
            # 对 prev_frame 和 next_frame 的插值帧进行加权平均
            interpolated_frame = cv2.addWeighted(interpolated_frame_from_prev, 0.5, interpolated_frame_from_next, 0.5,
                                                 0)
            return interpolated_frame

        interpolated_frames = []
        for i in range(len(sample_np) - 1):
            prev_frame = sample_np[i]
            next_frame = sample_np[i + 1]
            flow = calculate_optical_flow(prev_frame, next_frame)
            interpolated_frame = generate_interpolated_frame(prev_frame, next_frame, flow)
            interpolated_frames.extend([prev_frame, interpolated_frame])
        # 添加最后一帧
        interpolated_frames.append(sample_np[-1])
        # 如果需要，将处理后的帧转换回PyTorch张量
        interpolated_frames_np = np.array(interpolated_frames)
        interpolated_frames_tensor = torch.from_numpy(interpolated_frames_np).float()
        # 帧数  高 宽 通道
        # 调整维度顺序以匹配 [通道数, 新的帧数, 高度, 宽度]
        interpolated_frames_tensor = interpolated_frames_tensor.permute(3, 0, 1, 2)
        # 添加批次大小维度，最终形状为 [1, 3, 新的帧数, 384, 384]
        sample = interpolated_frames_tensor.unsqueeze(0)

        save_sample_path = os.path.join(self.savedir_sample, f"{sample_idx}.mp4")
        save_videos_grid(sample, save_sample_path)

        sample_config = {
            "prompt": prompt_textbox,
             #"n_prompt": negative_prompt_textbox,
            "sampler": "Euler",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "width": 384,
            "height": 384,
            "video_length": length_slider,
            "seed": seed
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")

        return gr.Video.update(value=save_sample_path)


controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
        """
        <div class="markdown-title">
        <h1>ConCLVD: Controllable Chinese Landscape Video Generation via Diffusion Model</h1>
        </div>
        """,
        classname="markdown-title"
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Select Model(Please select in order).
                """
            )
            with gr.Row():
                with gr.Column(scale=2):
                    stable_diffusion_dropdown = gr.Dropdown(
                    label="Pretrained Model Path",
                    choices=controller.stable_diffusion_list,
                    interactive=True,
            )
                stable_diffusion_dropdown.change(fn=controller.update_stable_diffusion,
                                                 inputs=[stable_diffusion_dropdown],
                                                 outputs=[stable_diffusion_dropdown])

                with gr.Column(scale=2):
                    motion_module_dropdown = gr.Dropdown(
                    label="Select motion module",
                    choices=controller.motion_module_list,
                    interactive=True,
                )
                motion_module_dropdown.change(fn=controller.update_motion_module, inputs=[motion_module_dropdown],
                                              outputs=[motion_module_dropdown])

                with gr.Column(scale=2):
                    base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model (required)",
                    choices=controller.personalized_model_list,
                    interactive=True,
                )
                    base_model_dropdown.change(fn=controller.update_base_model, inputs=[base_model_dropdown],
                                           outputs=[base_model_dropdown])
                '''
                lora_model_dropdown = gr.Dropdown(
                    label="Select LoRA model (optional)",
                    choices=["none"] + controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )
                
                lora_model_dropdown.change(fn=controller.update_lora_model, inputs=[lora_model_dropdown],
                                           outputs=[lora_model_dropdown])
                '''
                #lora_alpha_slider = gr.Slider(label="LoRA alpha", value=0.8, minimum=0, maximum=2, interactive=True)

                #personalized_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")

                def update_personalized_model():
                    controller.refresh_personalized_model()
                    return [
                        gr.Dropdown.update(choices=controller.personalized_model_list),
                        gr.Dropdown.update(choices=["none"] + controller.personalized_model_list)
                    ]
                '''
                personalized_refresh_button.click(fn=update_personalized_model, inputs=[],
                                                  outputs=[base_model_dropdown, lora_model_dropdown])
                '''
                #personalized_refresh_button.click(fn=update_personalized_model, inputs=[],outputs=[base_model_dropdown])

            with gr.Column(variant="panel"):
                gr.Markdown("### 2. Configs for ConCLVD.")
            
                # 定义下拉菜单选项，包括两个预设选项和一个'Custom'选项
                preset_prompts = ["There is a withered tree on the mountaintop.there is another branch with petals falling. Across from it is still a mountain.Birds fly by in the sky.", "A person drives a small boat on a river in the mountains.some vegetation on both sides of the lake, a red-crowned crane flying over the lake", "Enter Your Own Description"]
                prompt_dropdown = gr.Dropdown(label="Choose or enter a prompt", choices=preset_prompts)
            
                # 当用户选择'Custom'时，允许输入自定义文本
                prompt_textbox = gr.Textbox(label="Enter your prompt", visible=False, lines=2)

                # 根据用户的选择显示或隐藏文本框
                def prompt_input_handler(choice):
                    if choice == "Enter Your Own Description":
                        return gr.Textbox.update(visible=True, value="")
                    else:
                        return gr.Textbox.update(visible=False, value=choice)

                prompt_dropdown.change(prompt_input_handler, inputs=[prompt_dropdown], outputs=[prompt_textbox])

                #negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2)

            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        pass
                        #sampler_dropdown = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        #sample_step_slider = gr.Slider(label="Sampling steps", value=25, minimum=10, maximum=100,step=1)

                    #width_slider = gr.Slider(label="Width", value=512, minimum=256, maximum=1024, step=64)
                    #height_slider = gr.Slider(label="Height", value=512, minimum=256, maximum=1024, step=64)
                    length_slider = gr.Slider(label="Animation length", value=16, minimum=8, maximum=24, step=1)
                    #cfg_scale_slider = gr.Slider(label="CFG Scale", value=7.5, minimum=0, maximum=20)

                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)), inputs=[],
                                          outputs=[seed_textbox])
                
                    generate_button = gr.Button(value="Generate", variant='primary')

                result_video = gr.Video(label="Generated Animation", interactive=False)
            

            generate_button.click(
                fn=controller.animate,
                inputs=[
                    stable_diffusion_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    # 使用 prompt_textbox 的值作为输入
                    prompt_textbox,
                    length_slider,
                    seed_textbox,
                ],
                outputs=[result_video]
            )

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.launch(share=True)