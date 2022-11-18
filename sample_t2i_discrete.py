import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import utils
from datasets import get_dataset
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import builtins
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info')
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)

    with open(config.input_path, 'r') as f:
        prompts = f.read().strip().split('\n')

    print(prompts)

    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    contexts = clip.encode(prompts)

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    def cfg_nnet(x, timesteps, context):
        _cond = nnet(x, timesteps, context=context)
        if config.sample.scale == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + config.sample.scale * (_cond - _uncond)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    logging.info(config.sample)
    logging.info(f'mixed_precision={config.mixed_precision}')
    logging.info(f'N={N}')

    z_init = torch.randn(contexts.size(0), *config.z_shape, device=device)
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

    def model_fn(x, t_continuous):
        t = t_continuous * N
        return cfg_nnet(x, t, context=contexts)

    dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
    z = dpm_solver.sample(z_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
    samples = dataset.unpreprocess(decode(z))

    os.makedirs(config.output_path, exist_ok=True)
    for sample, prompt in zip(samples, prompts):
        save_image(sample, os.path.join(config.output_path, f"{prompt}.png"))



from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output images.")
flags.DEFINE_string("input_path", None, "The path to input texts.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.input_path = FLAGS.input_path
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
