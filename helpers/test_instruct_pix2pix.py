import PIL
import requests
import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
import gradio as gr

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


def generate(input_image, instruction, steps, image_guidance, text_guidance):
    image = pipe(
        instruction,
        image=input_image,
        num_inference_steps=steps,
        image_guidance_scale=image_guidance,
        guidance_scale=text_guidance,
    ).images[0]
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            instruction = gr.Textbox(
                label="Instruction", placeholder="e.g. turn him into a cyborg"
            )
            steps = gr.Slider(
                label="Inference Steps", minimum=1, maximum=100, value=20, step=1
            )
            image_guidance = gr.Slider(
                label="Image Guidance Scale", minimum=1, maximum=5, value=1.5, step=0.1
            )
            text_guidance = gr.Slider(
                label="Text Guidance Scale", minimum=1, maximum=20, value=7.5, step=0.5
            )
            submit_btn = gr.Button("Generate")
        with gr.Column():
            output_image = gr.Image(label="Output Image")

    submit_btn.click(
        fn=generate,
        inputs=[input_image, instruction, steps, image_guidance, text_guidance],
        outputs=output_image,
    )

demo.launch(share=True)
