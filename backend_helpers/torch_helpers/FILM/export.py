import warnings

import numpy as np
import tensorflow as tf
import torch

from interpolator import Interpolator


def translate_state_dict(var_dict, state_dict):
    for name, (prev_name, weight) in zip(state_dict, var_dict.items()):
        print('Mapping', prev_name, '->', name)
        weight = torch.from_numpy(weight)
        if 'kernel' in prev_name:
            # Transpose the conv2d kernel weights, since TF uses (H, W, C, K) and PyTorch uses (K, C, H, W)
            weight = weight.permute(3, 2, 0, 1)

        assert state_dict[name].shape == weight.shape, f'Shape mismatch {state_dict[name].shape} != {weight.shape}'

        state_dict[name] = weight


def import_state_dict(interpolator: Interpolator, saved_model):
    variables = saved_model.keras_api.variables

    extract_dict = interpolator.extract.state_dict()
    flow_dict = interpolator.predict_flow.state_dict()
    fuse_dict = interpolator.fuse.state_dict()

    extract_vars = {}
    _flow_vars = {}
    _fuse_vars = {}

    for var in variables:
        name = var.name
        if name.startswith('feat_net'):
            extract_vars[name[9:]] = var.numpy()
        elif name.startswith('predict_flow'):
            _flow_vars[name[13:]] = var.numpy()
        elif name.startswith('fusion'):
            _fuse_vars[name[7:]] = var.numpy()

    # reverse order of modules to allow jit export
    # TODO: improve this hack
    flow_vars = dict(sorted(_flow_vars.items(), key=lambda x: x[0].split('/')[0], reverse=True))
    fuse_vars = dict(sorted(_fuse_vars.items(), key=lambda x: int((x[0].split('/')[0].split('_')[1:] or [0])[0]) // 3, reverse=True))

    assert len(extract_vars) == len(extract_dict), f'{len(extract_vars)} != {len(extract_dict)}'
    assert len(flow_vars) == len(flow_dict), f'{len(flow_vars)} != {len(flow_dict)}'
    assert len(fuse_vars) == len(fuse_dict), f'{len(fuse_vars)} != {len(fuse_dict)}'

    for state_dict, var_dict in ((extract_dict, extract_vars), (flow_dict, flow_vars), (fuse_dict, fuse_vars)):
        translate_state_dict(var_dict, state_dict)

    interpolator.extract.load_state_dict(extract_dict)
    interpolator.predict_flow.load_state_dict(flow_dict)
    interpolator.fuse.load_state_dict(fuse_dict)


def verify_debug_outputs(pt_outputs, tf_outputs):
    max_error = 0
    for name, predicted in pt_outputs.items():
        if name == 'image':
            continue
        pred_frfp = [f.permute(0, 2, 3, 1).detach().cpu().numpy() for f in predicted]
        true_frfp = [f.numpy() for f in tf_outputs[name]]

        for i, (pred, true) in enumerate(zip(pred_frfp, true_frfp)):
            assert pred.shape == true.shape, f'{name} {i} shape mismatch {pred.shape} != {true.shape}'
            error = np.max(np.abs(pred - true))
            max_error = max(max_error, error)
            assert error < 1, f'{name} {i} max error: {error}'
    print('Max intermediate error:', max_error)


def test_model(interpolator, model, half=False, gpu=False):
    torch.manual_seed(0)
    time = torch.full((1, 1), .5)
    x0 = torch.rand(1, 3, 256, 256)
    x1 = torch.rand(1, 3, 256, 256)

    x0_ = tf.convert_to_tensor(x0.permute(0, 2, 3, 1).numpy(), dtype=tf.float32)
    x1_ = tf.convert_to_tensor(x1.permute(0, 2, 3, 1).numpy(), dtype=tf.float32)
    time_ = tf.convert_to_tensor(time.numpy(), dtype=tf.float32)
    tf_outputs = model({'x0': x0_, 'x1': x1_, 'time': time_}, training=False)

    if half:
        x0 = x0.half()
        x1 = x1.half()
        time = time.half()

    if gpu and torch.cuda.is_available():
        x0 = x0.cuda()
        x1 = x1.cuda()
        time = time.cuda()

    with torch.no_grad():
        pt_outputs = interpolator.debug_forward(x0, x1, time)

    verify_debug_outputs(pt_outputs, tf_outputs)

    with torch.no_grad():
        prediction = interpolator(x0, x1, time)
    output_color = prediction.permute(0, 2, 3, 1).detach().cpu().numpy()
    true_color = tf_outputs['image'].numpy()
    error = np.abs(output_color - true_color).max()

    print('Color max error:', error)


def main(model_path, save_path, export_to_torchscript=True, use_gpu=False, fp16=True, skiptest=False):
    print(f'Exporting model to FP{["32", "16"][fp16]} {["state_dict", "torchscript"][export_to_torchscript]} '
          f'using {"CG"[use_gpu]}PU')
    model = tf.compat.v2.saved_model.load(model_path)
    interpolator = Interpolator()
    interpolator.eval()
    import_state_dict(interpolator, model)

    if use_gpu and torch.cuda.is_available():
        interpolator = interpolator.cuda()
    else:
        if fp16 and use_gpu:
            warnings.warn('No GPU is available, using CPU FP32', UserWarning)
        fp16 = False

    if fp16:
        interpolator = interpolator.half()
    if export_to_torchscript:
        interpolator = torch.jit.script(interpolator)

    if not skiptest:
        test_model(interpolator, model, fp16, use_gpu)

    if export_to_torchscript:
        interpolator.save(save_path)
    else:
        torch.save(interpolator.state_dict(), save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Export frame-interpolator model to PyTorch state dict')

    parser.add_argument('model_path', type=str, help='Path to the TF SavedModel')
    parser.add_argument('save_path', type=str, help='Path to save the PyTorch state dict')
    parser.add_argument('--statedict', action='store_true', help='Export to state dict instead of TorchScript')
    parser.add_argument('--fp32', action='store_true', help='Save at full precision')
    parser.add_argument('--skiptest', action='store_true', help='Skip testing and save model immediately instead')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')

    args = parser.parse_args()

    main(args.model_path, args.save_path, not args.statedict, args.gpu, not args.fp32, args.skiptest)
