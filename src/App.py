from src.Replacer import Replacer
import gradio as gr


class App:
    def __init__(self):
        
        self._replacer = Replacer()
        print("replacer setup complete")
        

    def _process(self, 
                input_image, 
                prompt, 
                a_prompt, 
                n_prompt, 
                num_samples, 
                image_resolution, 
                ddim_steps, 
                guess_mode, 
                strength, 
                scale, 
                seed, 
                eta, 
                mask_blur):
        
        result = self._replacer.replace(
                input_image, 
                prompt,
                a_prompt, 
                n_prompt, 
                num_samples, 
                image_resolution, 
                ddim_steps, 
                guess_mode, 
                strength, 
                scale, 
                seed, 
                eta, 
                mask_blur)

        return result

        

    def run(self):
        block = gr.Blocks().queue()
        with block:
            with gr.Row():
                gr.Markdown("## Replacer")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="pil")
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
                    mask_blur = gr.Slider(label="Mask Blur", minimum=0.1, maximum=7.0, value=5.0, step=0.01)
                    with gr.Accordion("Advanced options", open=False):
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mask_blur]
            run_button.click(fn=self._process, inputs=ips, outputs=[result_gallery])


        block.launch(server_name='0.0.0.0')