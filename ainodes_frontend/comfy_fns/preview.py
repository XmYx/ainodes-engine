"""Step-preview callback registry for samplers.

ComfyUI's own ``latent_preview`` module decodes previews and pushes them over
the websocket to the browser.  aiNodes has no websocket - the sampler nodes
want to paint straight into their own widget - so they register a callback
here and the sampler passes it to ``comfy.sample``.
"""


class Previewer:
    def __init__(self):
        self.callback = None

    def set_callback(self, callback):
        self.callback = callback

    def prepare_callback(self, model=None, steps=0, x0_output_dict=None):
        """Same signature as ``latent_preview.prepare_callback``."""
        return self.callback


#: Module-level instance the sampler nodes share.
latent_preview = Previewer()
