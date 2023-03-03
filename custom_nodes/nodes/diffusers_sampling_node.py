import torch

from custom_nodes.auto_base_node import AutoBaseNode
from custom_nodes.base_widgets.diffusers_base import DiffusersSamplingBaseWidget

import singleton

gs = singleton.Singleton()
class DiffusersSamplingNode(AutoBaseNode):
    """
    An example of a node with a embedded QLineEdit.
    """

    # unique node identifier.
    __identifier__ = 'nodes.widget'

    # initial default node name.
    NODE_NAME = 'diffusers_sampling'
    def __init__(self, parent=None):
        super(DiffusersSamplingNode, self).__init__()
        #self.custom = DiffusersSamplingBaseWidget(self.view, self)
        #self.add_custom_widget(self.custom, tab='Custom')

        self.add_input("prompt_embeds_exe")
        self.add_input("pipe")
        self.add_output("out_image")

        self.create_property("prompt_embeds_exe", None)
        self.create_property("pipe", None)
        self.create_property("out_image", None)

    @torch.no_grad()
    def execute(self):

        num_inference_steps = 10
        prompt_embeds = self.get_property('prompt_embeds_exe')
        pipe = self.get_property("pipe")
        #print("prompt_embeds_type", type(prompt_embeds))


        generator = torch.Generator()
        latents = None
        do_classifier_free_guidance = True
        cross_attention_kwargs = None
        guidance_scale = 7.5
        eta = 0.0
        callback_steps = 1
        callback = None
        output_type = 'pil'
        #pipe = self.get_property('pipe')
        device = gs.obj[pipe].device

        # 4. Prepare timesteps
        gs.obj[pipe].scheduler.set_timesteps(num_inference_steps, device=gs.obj[pipe].device)
        timesteps = gs.obj[pipe].scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = gs.obj[pipe].unet.in_channels
        latents = gs.obj[pipe].prepare_latents(
            1 * 1,
            num_channels_latents,
            512,
            512,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the gs.obj[pipe]line
        extra_step_kwargs = gs.obj[pipe].prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * gs.obj[pipe].scheduler.order
        with gs.obj[pipe].progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = gs.obj[pipe].scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = gs.obj[pipe].unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = gs.obj[pipe].scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % gs.obj[pipe].scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = gs.obj[pipe].decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = gs.obj[pipe].run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = gs.obj[pipe].numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = gs.obj[pipe].decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = gs.obj[pipe].run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(gs.obj[pipe], "final_offload_hook") and gs.obj[pipe].final_offload_hook is not None:
            gs.obj[pipe].final_offload_hook.offload()

        self.set_property('out_image', image[0], push_undo=False)
        self.execute_children()
        #super().execute()
        return image

