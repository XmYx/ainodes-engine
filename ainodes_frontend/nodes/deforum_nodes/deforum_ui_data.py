
deforum_base_params = {

    "W": {
        "type": "spinbox",
        "default": 512,
        "min": 64,
        "max": 4096,
        "step": 64
    },
    "H": {
        "type": "spinbox",
        "default": 512,
        "min": 64,
        "max": 4096,
        "step": 64
    },
    "seed": {
        "type": "lineedit",
        "default": "-1",
    },
    "sampler": {
        "type": "lineedit",
        "default": "euler_ancestral"
    },
    "steps": {
        "type": "spinbox",
        "default": 25,
        "min": 0,
        "max": 10000,
        "step": 1
    },
    "scale": {
        "type": "spinbox",
        "default": 7,
        "min": 0,
        "max": 10000,
        "step": 1
    },
    "n_batch": {
        "type": "spinbox",
        "default": 1,
        "min": 0,
        "max": 10000,
        "step": 1
    },
    "batch_name": {
        "type": "lineedit",
        "default": "Deforum_{timestring}"
    },
    "seed_behavior": {
        "type": "dropdown",
        "choices": ["iter", "fixed", "random", "ladder", "alternate", "schedule"]
    },
    "seed_iter_N": {
        "type": "spinbox",
        "default": 1,
        "min": 0,
        "max": 10000,
        "step": 1
    },
    "outdir": {
        "type": "lineedit",
        "default": "output/deforum"
    },
    "strength": {
        "type": "doublespinbox",
        "default": 0.8,
        "min": 0,
        "max": 1,
        "step": 0.01
    },
    "save_settings": {
        "type": "checkbox",
        "default": True
    },
    "save_sample_per_step": {
        "type": "checkbox",
        "default": False
    },
    "prompt_weighting": {
        "type": "checkbox",
        "default": False
    },
    "normalize_prompt_weights": {
        "type": "checkbox",
        "default": True
    },
    "log_weighted_subprompts": {
        "type": "checkbox",
        "default": False
    },
    "prompt": {
        "type": "lineedit",
        "default": ""
    },
    "timestring": {
        "type": "lineedit",
        "default": ""
    },
    "seed_internal": {
        "type": "lineedit",
        "default": "-1",
    },


}


deforum_anim_params = {

    "animation_mode": {
        "type": "dropdown",
        "choices": ["None", "2D", "3D", "Video Input", "Interpolation"]
    },
    "max_frames": {
        "type": "spinbox",
        "min": 1,
        "max": 100000,
        "default": 120,
        "step": 1
    },
    "border": {
        "type": "dropdown",
        "choices": ["wrap", "replicate", "zeros"]
    },
    "resume_from_timestring": {
        "type": "checkbox",
        "default": False
    },
    "resume_timestring": {
        "type": "lineedit",
        "default": "20230129210106"
    },
    "use_looper": {
        "type": "checkbox",
        "default": False
    },

}


deforum_translation_params = {

    "angle": {
        "type": "lineedit",
        "default": "0:(0)"
    },
    "zoom": {
        "type": "lineedit",
        "default": "0:(1.0025+0.002*sin(1.25*3.14*t/30))"
    },
    "translation_x": {
        "type": "lineedit",
        "default": "0:(0)"
    },
    "translation_y": {
        "type": "lineedit",
        "default": "0:(0)"
    },
    "translation_z": {
        "type": "lineedit",
        "default": "0:(1.75)"
    },
    "transform_center_x": {
        "type": "lineedit",
        "default": "0:(0.5)"
    },
    "transform_center_y": {
        "type": "lineedit",
        "default": "0:(0.5)"
    },
    "rotation_3d_x": {
        "type": "lineedit",
        "default": "0:(0)"
    },
    "rotation_3d_y": {
        "type": "lineedit",
        "default": "0:(0)"
    },
    "rotation_3d_z": {
        "type": "lineedit",
        "default": "0:(0)"
    },

}

deforum_hybrid_video_params = {

    "hybrid_generate_inputframes": {
        "type": "checkbox",
        "default": False
    },
    "hybrid_generate_human_masks": {
        "type": "dropdown",
        "choices": ["None", "PNGs", "Video", "Both"],
        "default": "None"
    },
    "hybrid_use_first_frame_as_init_image": {
        "type": "checkbox",
        "default": True
    },
    "hybrid_motion": {
        "type": "dropdown",
        "choices": ["None", "Optical Flow", "Perspective", "Affine"],
        "default": "None"
    },
    "hybrid_motion_use_prev_img": {
        "type": "checkbox",
        "default": False
    },
    "hybrid_flow_consistency": {
        "type": "checkbox",
        "default": False
    },
    "hybrid_consistency_blur": {
        "type": "spinbox",
        "min": 0,
        "max": 10,
        "default": 2,
        "step": 1
    },
    "hybrid_flow_method": {
        "type": "dropdown",
        "choices": ["RAFT", "DIS Medium", "DIS Fine", "Farneback"],
        "default": "RAFT"
    },
    "hybrid_composite": {
        "type": "dropdown",
        "choices": ["None", "Normal", "Before Motion", "After Generation"],
        "default": "None"
    },
    "hybrid_use_init_image": {
        "type": "checkbox",
        "default": False
    },
    "hybrid_comp_mask_type": {
        "type": "dropdown",
        "choices": ["None", "Depth", "Video Depth", "Blend", "Difference"],
        "default": "None"
    },
    "hybrid_comp_mask_inverse": {
        "type": "checkbox",
        "default": False
    },
    "hybrid_comp_mask_equalize": {
        "type": "dropdown",
        "choices": ["None", "Before", "After", "Both"],
        "default": "None"
    },
    "hybrid_comp_mask_auto_contrast": {
        "type": "checkbox",
        "default": False
    },
    "hybrid_comp_save_extra_frames": {
        "type": "checkbox",
        "default": False
    },

}

deforum_hybrid_video_schedules = {

    "hybrid_comp_alpha_schedule": {
        "type": "lineedit",
        "default": "0:(0.5)"
    },
    "hybrid_comp_mask_blend_alpha_schedule": {
        "type": "lineedit",
        "default": "0:(0.5)"
    },
    "hybrid_comp_mask_contrast_schedule": {
        "type": "lineedit",
        "default": "0:(1)"
    },
    "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": {
        "type": "lineedit",
        "default": "0:(100)"
    },
    "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": {
        "type": "lineedit",
        "default": "0:(0)"
    },
    "hybrid_flow_factor_schedule": {
        "type": "lineedit",
        "default": "0:(1)"
    },

}

deforum_color_coherence_params = {

    "color_coherence": {
        "type": "dropdown",
        "choices": ["None", "HSV", "LAB", "RGB", "Video Input", "Image"]
    },
    "color_coherence_image_path": {
        "type": "lineedit",
        "default": ""
    },
    "color_coherence_video_every_N_frames": {
        "type": "spinbox",
        "min": 1,
        "max": 100,
        "default": 1,
        "step": 1
    },
    "color_force_grayscale": {
        "type": "checkbox",
        "default": False
    },
    "legacy_colormatch": {
        "type": "checkbox",
        "default": False
    },

}


deforum_depth_params = {

    "use_depth_warping": {
        "type": "checkbox",
        "default": True
    },
    "depth_algorithm": {
        "type": "dropdown",
        "choices": [
            "Zoe",
            "Midas-3-Hybrid",
            "Midas+AdaBins (old)",
            "Zoe+AdaBins (old)",
            "Midas-3.1-BeitLarge",
            "AdaBins",
            "Leres"
        ],
        "default": "Zoe"
    },
    "midas_weight": {
        "type": "doublespinbox",
        "min": 0,
        "max": 1,
        "default": 0.2,
        "step": 0.01
    },
    "padding_mode": {
        "type": "dropdown",
        "choices": ["border", "reflection", "zeros"],
        "default": "border"
    },
    "sampling_mode": {
        "type": "dropdown",
        "choices": ["bicubic", "bilinear", "nearest"],
        "default": "bicubic"
    },
    "save_depth_maps": {
        "type": "checkbox",
        "default": False
    },

}

deforum_cadence_params = {

    "diffusion_cadence": {
        "type": "spinbox",
        "min": 0,
        "max": 100,
        "default": 4,
        "step": 1
    },
    "optical_flow_cadence": {
        "type": "dropdown",
        "choices": ["None", "RAFT", "DIS Medium", "DIS Fine", "Farneback"]
    },
    "cadence_flow_factor_schedule": {
        "type": "lineedit",
        "default": "0: (1)"
    },
    "optical_flow_redo_generation": {
        "type": "dropdown",
        "choices": ["None", "RAFT", "DIS Medium", "DIS Fine", "Farneback"]
    },
    "redo_flow_factor_schedule": {
        "type": "lineedit",
        "default": "0: (1)"
    },
    "diffusion_redo": {
        "type": "lineedit",
        "default": "0"
    },

}


deforum_video_init_params = {

    "video_init_path": {
        "type": "lineedit",
        "default": "https://deforum.github.io/a1/V1.mp4"
    },
    "extract_nth_frame": {
        "type": "spinbox",
        "min": 1,
        "max": 100,
        "default": 1,
        "step": 1
    },
    "extract_from_frame": {
        "type": "spinbox",
        "min": 0,
        "max": 1000,
        "default": 0,
        "step": 1
    },
    "extract_to_frame": {
        "type": "spinbox",
        "min": -1,
        "max": 1000,
        "default": -1,
        "step": 1
    },
    "overwrite_extracted_frames": {
        "type": "checkbox",
        "default": True
    },
    "use_mask_video": {
        "type": "checkbox",
        "default": False
    },
    "video_mask_path": {
        "type": "lineedit",
        "default": "https://deforum.github.io/a1/VM1.mp4"
    },
}


deforum_masking_params = {

    "use_mask": {
        "type": "checkbox",
        "default": False
    },
    "use_alpha_as_mask": {
        "type": "checkbox",
        "default": False
    },
    "mask_file": {
        "type": "lineedit",
        "default": "https://deforum.github.io/a1/M1.jpg"
    },
    "mask_image": {
        "type": "lineedit",
        "default": None
    },
    "noise_mask": {
        "type": "lineedit",
        "default": None
    },

    "invert_mask": {
        "type": "checkbox",
        "default": False
    },
    "mask_contrast_adjust": {
        "type": "doublespinbox",
        "default": 1.0,
        "min": 0,
        "max": 10,
        "step": 0.1
    },
    "mask_brightness_adjust": {
        "type": "doublespinbox",
        "default": 1.0,
        "min": 0,
        "max": 10,
        "step": 0.1
    },
    "overlay_mask": {
        "type": "checkbox",
        "default": True
    },
    "mask_overlay_blur": {
        "type": "spinbox",
        "default": 4,
        "min": 0,
        "max": 100,
        "step": 1
    },
    "fill": {
        "type": "spinbox",
        "default": 1,
        "min": 0,
        "max": 10000,
        "step": 1
    },
    "full_res_mask": {
        "type": "checkbox",
        "default": True
    },
    "full_res_mask_padding": {
        "type": "spinbox",
        "default": 4,
        "min": 0,
        "max": 10000,
        "step": 1
    },

}

deforum_diffusion_schedule_params = {

    "noise_schedule": {
        "type": "lineedit",
        "default": "0: (0.065)"
    },
    "strength_schedule": {
        "type": "lineedit",
        "default": "0: (0.65)"
    },
    "contrast_schedule": {
        "type": "lineedit",
        "default": "0: (1.0)"
    },
    "cfg_scale_schedule": {
        "type": "lineedit",
        "default": "0: (7)"
    },
    "enable_steps_scheduling": {
        "type": "checkbox",
        "default": False
    },
    "steps_schedule": {
        "type": "lineedit",
        "default": "0: (25)"
    },
    "enable_ddim_eta_scheduling": {
        "type": "checkbox",
        "default": False
    },
    "ddim_eta_schedule": {
        "type": "lineedit",
        "default": "0:(0)"
    },
    "enable_ancestral_eta_scheduling": {
        "type": "checkbox",
        "default": False
    },
    "ancestral_eta_schedule": {
        "type": "lineedit",
        "default": "0:(1)"
    }

}

deforum_noise_params = {

    "enable_noise_multiplier_scheduling": {
        "type": "checkbox",
        "default": True
    },
    "noise_multiplier_schedule": {
        "type": "lineedit",
        "default": "0: (1.05)"
    },
    "amount_schedule": {
        "type": "lineedit",
        "default": "0: (0.1)"
    },
    "kernel_schedule": {
        "type": "lineedit",
        "default": "0: (5)"
    },
    "sigma_schedule": {
        "type": "lineedit",
        "default": "0: (1.0)"
    },
    "threshold_schedule": {
        "type": "lineedit",
        "default": "0: (0.0)"
    },
    "noise_type": {
        "type": "dropdown",
        "choices": ["uniform", "perlin"]
    },
    "perlin_w": {
        "type": "spinbox",
        "min": 1,
        "max": 100,
        "default": 8,
        "step": 1
    },
    "perlin_h": {
        "type": "spinbox",
        "min": 1,
        "max": 100,
        "default": 8,
        "step": 1
    },
    "perlin_octaves": {
        "type": "spinbox",
        "min": 1,
        "max": 10,
        "default": 4,
        "step": 1
    },
    "perlin_persistence": {
        "type": "doublespinbox",
        "min": 0.1,
        "max": 1.0,
        "default": 0.5,
        "step": 0.1
    },

}


deforum_args_layout = {
                "enable_perspective_flip": {
                    "type": "checkbox",
                    "default": False
                },
                "perspective_flip_theta": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "perspective_flip_phi": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "perspective_flip_gamma": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "perspective_flip_fv": {
                    "type": "lineedit",
                    "default": "0:(53)"
                },
                "fov_schedule": {
                    "type": "lineedit",
                    "default": "0: (70)"
                },
                "aspect_ratio_schedule": {
                    "type": "lineedit",
                    "default": "0: (1)"
                },
                "aspect_ratio_use_old_formula": {
                    "type": "checkbox",
                    "default": False
                },
                "near_schedule": {
                    "type": "lineedit",
                    "default": "0: (200)"
                },
                "far_schedule": {
                    "type": "lineedit",
                    "default": "0: (10000)"
                },
                "seed_schedule": {
                    "type": "lineedit",
                    "default": "0:(s), 1:(-1), \"max_f-2\":(-1), \"max_f-1\":(s)"
                },
                "pix2pix_img_cfg_scale": {
                    "type": "doublespinbox",
                    "min": 0.0,
                    "max": 10.0,
                    "default": 1.5,
                    "step": 0.1
                },
                "pix2pix_img_cfg_scale_schedule": {
                    "type": "lineedit",
                    "default": "0:(1.5)"
                },
                "enable_subseed_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "subseed_schedule": {
                    "type": "lineedit",
                    "default": "0:(1)"
                },
                "subseed_strength_schedule": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "enable_sampler_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "sampler_schedule": {
                    "type": "lineedit",
                    "default": "0: (\"Euler a\")"
                },
                "use_noise_mask": {
                    "type": "checkbox",
                    "default": False
                },
                "mask_schedule": {
                    "type": "lineedit",
                    "default": "0: (\"{video_mask}\")"
                },
                "noise_mask_schedule": {
                    "type": "lineedit",
                    "default": "0: (\"{video_mask}\")"
                },
                "enable_checkpoint_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "checkpoint_schedule": {
                    "type": "lineedit",
                    "default": "0: (\"model1.ckpt\"), 100: (\"model2.safetensors\")"
                },
                "enable_clipskip_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "clipskip_schedule": {
                    "type": "lineedit",
                    "default": "0: (2)"
                },


        }



deforum_image_init_params = {

    "strength_0_no_init": {
      "type": "checkbox",
      "default": True
    },
    "init_image": {
      "type": "lineedit",
      "default": "https://deforum.github.io/a1/I1.png"
    },
    "init_sample": {
      "type": "lineedit",
      "default": None
    },
    "use_init": {
        "type": "checkbox",
        "default": False
    },

}

