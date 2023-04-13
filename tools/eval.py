import argparse
import collections
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from tqdm import tqdm
import model.models as module_arch
from configs.parse_config import ConfigParser

import os


def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, generator, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs)//batch):
            g_images.append(generator(zs[i*batch:(i+1)*batch]).cpu())
        if len(zs) % batch > 0:
            g_images.append(generator(zs[-(len(zs)%batch):]).cpu())
    return torch.cat(g_images)

def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), f'{folder_name}/{i}.jpg')


def main(config: ConfigParser):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model.DataParallel()
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.generator.eval()

    latent_dim = config['arch']['args']['latent_dim']

    # generate images
    with torch.no_grad():
        for i in tqdm(range(config['eval']['n_sample']//config['eval']['batch_size'])):
            noise = torch.randn(config['eval']['batch_size']    , latent_dim).to(device)
            generated_imgs = model.generator(noise)
            generated_imgs = F.interpolate(generated_imgs, 512)
            for j, g_img in enumerate( generated_imgs ):
                vutils.save_image(g_img.add(1).mul(0.5), 
                    os.path.join(config['eval']['save_dir'], '%d.png'%(i*config['eval']['batch_size']+j)))#, normalize=True, range=(-1,1))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Generating images from trained GAN model')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-dir', '--dist'], type=str, target='eval;save_dir'),
        CustomArgs(['-n', '--n_sample'], type=int, target='eval;n_sample'),
        CustomArgs(['-bs', '--batch_size'], type=int, target='eval;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    main(config)