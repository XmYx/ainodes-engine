#dummy file for comfyui nodes
import glob
import importlib
import time
import traceback

from ainodes_frontend.comfy_fns.adapter_nodes.adapter_utils import create_node



def common_ksampler_with_custom_noise(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
                                      denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                                      force_full_denoise=False, noise=None):

    print("USING AINODES SAMPLER")

    latent_image = latent["samples"]
    if noise is not None:
        rng_noise = noise.next().detach().cpu()
        noise = rng_noise.clone()
    else:
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            from comfy.sample import prepare_noise
            noise = prepare_noise(latent_image, seed, batch_inds)


    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = False

    from comfy.sample import sample as sample_k

    samples = sample_k(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step,
                                  last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                                  disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples

    return (out,)



# from ai_nodes.ainodes_engine_comfy_nodes.adapter_nodes.adapter_utils import parse_comfynode

WAS = "GREAT"

import os, sys

#sys.path.extend([os.path.join(os.getcwd(), "src", "ComfyUI")])
import sys

#import comfy.sample

#from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.k_sampler import sample as ainodes_sample

#comfy.sample.sample = ainodes_sample

# from ainodes_frontend.base import modelmanagement_hijack
# import comfy.model_management
#
# comfy.model_management.unet_inital_load_device = modelmanagement_hijack.unet_inital_load_device


# sys.modules['comfy.model_management'] = modelmanagement_hijack

def load_comfy_node(module_path):
    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
    try:
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        else:
            module_spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_path, "__init__.py"))
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:

            mappings = getattr(module, "NODE_CLASS_MAPPINGS")


            return mappings

            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            return True
        else:
            print(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS.")
            return None
    except Exception as e:
        print(traceback.format_exc())
        print(f"Cannot import {module_path} module for custom nodes:", e)
        return None

def load_comfy_nodes():
    node_paths = [os.path.join(os.getcwd(), "src/deforum/src/ComfyUI/custom_nodes"), os.path.join(os.getcwd(), "src/deforum/src/ComfyUI/comfy_extras")]

    #print("COMFY NODE PATHS", node_paths)
    mappings = []
    node_import_times = []
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py": continue
            if module_path.endswith(".disabled"): continue
            time_before = time.perf_counter()
            mapping = load_comfy_node(module_path)
            if mappings is not None:
                #print(mapping)
                mappings.append(mapping)
            node_import_times.append((time.perf_counter() - time_before, module_path, True))

    if len(node_import_times) > 0:
        print("\nImport times for custom nodes:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
    #import nodes
    #mappings.append(nodes.NODE_CLASS_MAPPINGS)
    return mappings


possible_ports = []
possible_ui_elements = []
possible_output_ports = []

directory = "src/deforum/src/ComfyUI/custom_nodes"
def get_node_parameters(node_class):

    global possible_ports
    global possible_ui_elements

    ordered_inputs = []
    ordered_outputs = []

    params = {"class":node_class}


    for key, value in node_class.INPUT_TYPES().items():
        for value_name, value_params in value.items():

            is_port = isinstance(value_params, str)

            if not is_port and len(value_params) == 1:

                if isinstance(value_params[0], str):

                    is_port = True

            if is_port:

                if isinstance(value_params, str):
                    n = value_params
                else:
                    n = value_params[0]

                if n not in possible_ports:
                    possible_ports.append(n)
            else:
                if value_name not in possible_ui_elements:
                    possible_ui_elements.append((value_name, value_params))


            #print(value_name, value_params, type(value_params), "len", len(value_params), "UI" if not is_port else "PORT")
            ordered_inputs.append((value_name, value_params, "UI" if not is_port else "PORT"))
    params["inputs"] = ordered_inputs
    if isinstance(node_class.RETURN_TYPES, tuple):
        for item in node_class.RETURN_TYPES:
            if item.upper() not in possible_output_ports:
                possible_output_ports.append(item.upper())
            ordered_outputs.append(item.upper())
    elif isinstance(node_class.RETURN_TYPES, str):
        if node_class.RETURN_TYPES.upper() not in possible_output_ports:
            possible_output_ports.append(node_class.RETURN_TYPES.upper())
        ordered_outputs.append(node_class.RETURN_TYPES.upper())


    #print("PARAMS", params)

    return ordered_inputs, ordered_outputs

# print(possible_ui_elements)

known_ports = {"LATENT":2,
               "CONDITIONING":3,
               "IMAGE":5,
               "MASK":5,
               "VAE":4,
               "CLIP":4,
               "MODEL":4,
               "CONTROL_NET":4,
               }

def parse_comfynode(node_name, node_class, category):
    node_content_class = node_name.lower().replace(" ", "_")
    # print(node_class.INPUT_TYPES())
    ordered_inputs, ordered_outputs = get_node_parameters(node_class)
    inputs = []
    input_names = []
    ui_inputs = []
    for i in ordered_inputs:
        if i[2] == "PORT":
            if i[1][0] in known_ports:
                inputs.append(known_ports[i[1][0]])
            else:
                inputs.append(7)
            input_names.append(i[0].upper())
        else:
            ui_inputs.append(i)

            #print("SEARCHING HERE", i)
            if len(list(i[1])) > 1:

                if "defaultBehavior" in i[1][1]:
                    inputs.append(7)
                    input_names.append(i[0].upper())

    inputs.append(1)
    input_names.append('EXEC')

    outputs = []
    output_names = []

    for i in ordered_outputs:
        if i in known_ports:
            outputs.append(known_ports[i])
        else:
            outputs.append(7)
        output_names.append(i)

    outputs.append(1)
    output_names.append('EXEC')
    #print("creating_node", node_class)
    node = create_node(node_class=node_class,
                       node_name=node_name,
                       ui_inputs=ui_inputs,
                       inputs=inputs,
                       input_names=input_names,
                       outputs=outputs,
                       output_names=output_names,
                       category_input=category)






#import nodes
#from custom_nodes.ComfyUI_Disco_Diffusion.nodes import NODE_CLASS_MAPPINGS

#print("Comfy Node Class Mappings", nodes.NODE_CLASS_MAPPINGS)



#node_class_mappings.append(nodes.NODE_CLASS_MAPPINGS)

# for mapping in node_class_mappings:
#     if mapping is not None:
#         #print(mapping)
# try:
class Previewer:

    def __init__(self):
        self.callback = None

    def set_callback(self, callback):
        self.callback = callback

    def prepare_callback(self, model, steps):
        return self.callback

latent_preview = Previewer()
sys.modules['latent_preview'] = latent_preview



import nodes
#nodes.common_ksampler = common_ksampler_with_custom_noise



for node_name, node_class in nodes.NODE_CLASS_MAPPINGS.items():
    parse_comfynode(node_name, node_class, "ComfyUI Base")

node_class_mappings = load_comfy_nodes()

for mapping in node_class_mappings:
    if mapping is not None:
        for node_name, node_class in mapping.items():
            parse_comfynode(node_name, node_class, "ComfyUI Extras")
# except:
#     print('No Comfy nodes found')




# print("PORTS", possible_ports)
# print("UI", possible_ui_elements)
# print("OUT", possible_output_ports)
