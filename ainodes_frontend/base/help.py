


def get_help():
    return ["aiNodes - Help",
            "Home - Show Search"
            "F1 - Hide / Show Help",
            "F2 - Hide / Show MiniMap",
            "F11 - FullScreen",
            "Esc - Hide / Show Nodes",
            "` - Hide / Show Console",
            "Make sure to connect execution\n"
            "lines, and if using subgraphs\n"
            "to include, and connect a \n"
            "subgraph input node and a \n"
            "subgraph output node within your\n"
            "subgraph. These provide the entry\n"
            "and exit points from subgraphs,\n"
            "and allow you to have complicated\n" 
            "pipelines hidden in a single node."]



node_attrs = {
    "hypernetwork_loader":{
        "help":"Hypernetworks are essentially small models\n"
               "that steer your generation in a specific visual\n"
               "direction. It may be used with Stable Diffusion,\n"
               "please place it somewhere between your Torch Loader\n"
               "and KSampler. You may use multiple Hypernetworks,\n"
               "to do so, just place them in series, and make sure\n"
               "to connect their execution lines."
    },
    "canvas_node":{
        "help":"Canvas node to use with Inpainting.\n"
               "Please use an Encode for Inpaint node\n"
               "in order to initiate inpainting with\n"
               "any model."

    },
    "colormatch_node":{
        "help":"ColorMatch Node\n\n"
               "Use the bottom input as your color reference\n"
               "and the top Image input as your input image."
    },
    "image_disbg_node":{
        "help":"BG Removal Node\n\n"
               "Simple background removal for any subject."
    },
    "image_blend_node":{
        "help":"Image Blend Node\n\n"
               "Use this node to composite or blend your images.\n"
               "By default, a simple blend happens with the given\n"
               "blend value, but you can set the composite mode to\n"
               "various styles of compositing, just play around and\n"
               "re-evaluate the node to see your changes."
    },
    "image_list_node":{
        "help":"Image List Node\n\n"
               "You may load a folder of images\n"
               "at each evaluation, it will jump to\n"
               "the next image, and set it's output to\n"
               "that image, which then may be used\n"
               "for further processing"
    },
    "image_op_node":{
        "help":"Image Operator Node\n\n"
               "This node provides access to the ControlNet\n"
               "preprocessors named the same as the models,\n"
               "use it before the Apply ControlNet node.\n"
               "It also serves as a basic Image Filter node."
    },
    "image_preview_node": {
        "help":"Image Preview Node\n\n"
               "Simple Image preview / save node.\n"
               "Additionally, you may include the NodeGraph\n"
               "in your saved images by ticking\n"
               "the 'Save Meta to PNG' checkbox.\n"
               "These then may be drag and dropped into\n"
               "aiNodes, and reopened to recreate the image."
    },
    "image_input_node":{
        "help":"Image Input Node\n\n"
               "Use this node to input Images or Videos.\n"
               "at each evaluation, the next frame will be\n"
               "emitted."


    }
}

def get_node_attr(content_label_obj_name, attr):
        if content_label_obj_name in node_attrs:
            if attr in node_attrs[content_label_obj_name]:
                return node_attrs[content_label_obj_name][attr]
        return None