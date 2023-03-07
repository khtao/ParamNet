import argparse
import itertools
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
import torch.nn.functional as F
from source.image_pool import ImagePool
from source.model import ResnetGenerator, ParamNet, NLayerDiscriminator, GANloss, init_weights, lr_warmup
from source.utils import infiniteloop, set_seed, ImageDataset, Visualizer
import ignite


def cal_psnr(img1, img2):
    img2 = torch.round(img2)
    img1 = torch.round(img1)
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(255 ** 2 / mse)


def convert_gray(image):
    return image[:, 0:1, :, :] * 0.299 + image[:, 1:2, :, :] * 0.587 + image[:, 2:3, :, :] * 0.114


def norm_layer(image):
    return (image - image.mean(dim=[1, 2, 3])) / image.std(dim=[1, 2, 3])


def test(model, test_dataloader):
    model.eval()
    total = len(test_dataloader)
    ssim_func = ignite.metrics.SSIM(data_range=255, device=torch.device('cuda:0'))
    ssim_source_func = ignite.metrics.SSIM(data_range=255, device=torch.device('cuda:0'))
    psnr_func = ignite.metrics.PSNR(data_range=255, device=torch.device('cuda:0'))
    for i, (source_image, target_image) in tqdm(enumerate(test_dataloader), total=total):
        with torch.no_grad():
            target_image = target_image.cuda()
            source_image = source_image.cuda()
            image_out = model(source_image)
            target_image = target_image.cuda()
            image_out = image_out * 127.5 + 127.5
            target_image = target_image * 127.5 + 127.5
            source_image = source_image * 127.5 + 127.5
            image_out = image_out.round()
            target_image = target_image.round()
            source_image = source_image.round()
            ssim_func.update([image_out, target_image])
            psnr_func.update([image_out, target_image])
            image_out_gray = convert_gray(image_out)
            source_image_gray = convert_gray(source_image)

            image_out_gray = torch.cat([image_out_gray, image_out_gray, image_out_gray], dim=1)
            source_image_gray = torch.cat([source_image_gray, source_image_gray, source_image_gray], dim=1)
            # print(source_image_gray.size())
            ssim_source_func.update([image_out_gray.round(), source_image_gray.round()])

    return {"psnr": float(psnr_func.compute()),
            "ssim": float(ssim_func.compute()),
            "ssim_source": float(ssim_source_func.compute())}


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def print_options(opt, mparser):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = mparser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(expr_dir, exist_ok=True)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.name))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def train(opt):
    dataset = ImageDataset(opt.train_dir_root, dir_A=opt.dir_A,
                           dir_B=opt.dir_B, align=False, imgsize=opt.train_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchsize,
        shuffle=True, num_workers=opt.nThreads,
        drop_last=True)
    net_G_A = ParamNet(backbone=opt.backbone, resample_size=opt.resample_size,
                       channels=opt.channels, layers=opt.n_layer).to(device)
    # print(net_G_A)
    net_G_B = ParamNet(backbone=opt.backbone, resample_size=opt.resample_size,
                       channels=opt.channels, layers=opt.n_layer).to(device)

    net_G_AA = ResnetGenerator(3, 3, 64, norm_layer=nn.InstanceNorm2d, n_blocks=6).to(device)
    net_G_BB = ResnetGenerator(3, 3, 64, norm_layer=nn.InstanceNorm2d, n_blocks=6).to(device)

    net_D_A = NLayerDiscriminator(3, 64, 3, norm_layer=nn.InstanceNorm2d).to(device)
    net_D_B = NLayerDiscriminator(3, 64, 3, norm_layer=nn.InstanceNorm2d).to(device)
    fake_A_pool = ImagePool(opt.pool_size)
    fake_B_pool = ImagePool(opt.pool_size)
    init_weights(net_D_A, opt.init_type, opt.init_gain)
    init_weights(net_D_B, opt.init_type, opt.init_gain)
    init_weights(net_G_AA, opt.init_type, opt.init_gain)
    init_weights(net_G_BB, opt.init_type, opt.init_gain)
    init_weights(net_G_A, opt.init_type, opt.init_gain)
    init_weights(net_G_B, opt.init_type, opt.init_gain)
    if opt.pretrained:
        weight = torch.load(opt.pretrained)
        net_G_A.load_state_dict(weight['net_G_A'])
        net_G_B.load_state_dict(weight['net_G_B'])
        net_G_AA.load_state_dict(weight['net_G_AA'])
        net_G_BB.load_state_dict(weight['net_G_BB'])
        print('load weight from ', opt.pretrained)

    loss_fn = GANloss()
    loss_l1 = torch.nn.L1Loss()
    loss_mse = torch.nn.MSELoss()

    optim_G = optim.Adam(itertools.chain(net_G_A.parameters(),
                                         net_G_B.parameters(),
                                         net_G_AA.parameters(),
                                         net_G_BB.parameters()
                                         ),
                         lr=opt.lr_G,
                         betas=opt.betas)
    optim_D = optim.Adam(itertools.chain(net_D_A.parameters(),
                                         net_D_B.parameters()),
                         lr=opt.lr_D,
                         betas=opt.betas)
    step_num = opt.total_steps - opt.warmup_step
    # lf = muti_cycle([int(step_num * 0.5), int(step_num * 0.7), int(step_num * 0.9)], steps=step_num)
    sched_G = optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=lambda x: 1 - x / step_num)
    sched_D = optim.lr_scheduler.LambdaLR(optim_D, lr_lambda=lambda x: 1 - x / step_num)
    looper = infiniteloop(dataloader)
    vis = Visualizer(opt.name)
    step_now = 0
    best_psnr = 0
    fs = open(os.path.join(opt.checkpoints_dir, opt.name, opt.name + '_log.txt'), 'a')
    if opt.continue_train:
        weight = torch.load(os.path.join(opt.checkpoints_dir, opt.name, opt.name + '_last.pt'))
        net_G_A.load_state_dict(weight['net_G_A'])
        net_G_B.load_state_dict(weight['net_G_B'])
        net_G_AA.load_state_dict(weight['net_G_AA'])
        net_G_BB.load_state_dict(weight['net_G_BB'])
        net_D_A.load_state_dict(weight['net_D_A'])
        net_D_B.load_state_dict(weight['net_D_B'])
        sched_D.load_state_dict(weight['sched_D'])
        sched_G.load_state_dict(weight['sched_G'])
        optim_G.load_state_dict(weight['optim_G'])
        optim_D.load_state_dict(weight['optim_D'])
        if 'step' in weight.keys():
            opt.step_count = weight['step']
            step_now = weight['step']
        print('load from ', os.path.join(opt.checkpoints_dir, opt.name, opt.name + '_last.pt'))

        # for key in kk:
        #     mm = g(key)
    if opt.need_test==1:
        test_dataset = ImageDataset(opt.test_dir_root, dir_A=opt.dir_A,
                                    dir_B=opt.dir_B, align=True, imgsize=opt.test_size)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=opt.batchsize,
            shuffle=False, num_workers=opt.nThreads,
            drop_last=False)
        # mean_loss = test(net_G_A, test_dataloader)
        # print(mean_loss)
    last_cycle_loss = 1.0
    with trange(opt.step_count, opt.total_steps + 1, desc='Training', ncols=0) as pbar:
        for step in pbar:
            net_G_A.train()
            net_G_B.train()

            real_a, real_b = next(looper)
            real_a = real_a.to(device)
            real_b = real_b.to(device)
            if opt.random_scale:
                scale = random.randint(8, 32) * 8
                real_a = torch.nn.functional.interpolate(real_a, size=(scale, scale), mode='bilinear',
                                                         align_corners=True)
                real_b = torch.nn.functional.interpolate(real_b, size=(scale, scale), mode='bilinear',
                                                         align_corners=True)

            set_requires_grad([net_D_A, net_D_B], False)
            # Generator
            fake_b = net_G_A(real_a)
            fake_bb = net_G_AA(fake_b)
            fake_a = net_G_B(real_b)
            fake_aa = net_G_BB(fake_a)
            rec_b = net_G_A(fake_aa)
            rec_a = net_G_B(fake_bb)

            loss_g_a = loss_fn(net_D_A(fake_aa))
            loss_g_b = loss_fn(net_D_B(fake_bb))

            loss_cycle_a = loss_l1(rec_a, real_a) * opt.lambda_A
            loss_cycle_b = loss_l1(rec_b, real_b) * opt.lambda_B

            loss_G = loss_g_b + loss_g_a + loss_cycle_a + loss_cycle_b
            if opt.lambda_diff > 0:
                loss_diff_a = loss_mse(fake_a, fake_aa.detach()) * opt.lambda_diff
                loss_diff_b = loss_mse(fake_b, fake_bb.detach()) * opt.lambda_diff
                loss_G += loss_diff_a + loss_diff_b

            if opt.lambda_identity > 0:
                idt_b = net_G_A(real_b)
                idt_a = net_G_B(real_a)
                idt_bb = net_G_AA(real_b)
                idt_aa = net_G_BB(real_a)
                loss_idt_a = loss_l1(idt_aa, real_a) * opt.lambda_identity
                loss_idt_b = loss_l1(idt_bb, real_b) * opt.lambda_identity
                loss_idt_a += loss_l1(idt_a, real_a) * opt.lambda_identity
                loss_idt_b += loss_l1(idt_b, real_b) * opt.lambda_identity
                loss_G += loss_idt_b + loss_idt_a

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            set_requires_grad([net_D_A, net_D_B], True)
            net_D_A_real = net_D_A(real_a)
            image_d_aa = fake_A_pool.query(fake_aa.detach())
            net_D_A_fake = net_D_A(image_d_aa.detach())
            loss_d_a = loss_fn(net_D_A_real, net_D_A_fake)
            net_D_B_real = net_D_B(real_b)
            image_d_bb = fake_B_pool.query(fake_bb.detach())
            net_D_B_fake = net_D_B(image_d_bb.detach())
            loss_d_b = loss_fn(net_D_B_real, net_D_B_fake)
            loss_D = loss_d_a + loss_d_b

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
            if step % opt.display_freq == 0:
                vis.img('fake_b', fake_b[0] * 0.5 + 0.5)
                vis.img('fake_a', fake_a[0] * 0.5 + 0.5)
                vis.img('fake_bb', fake_bb[0] * 0.5 + 0.5)
                vis.img('fake_aa', fake_aa[0] * 0.5 + 0.5)
                vis.img('rec_a', rec_a[0] * 0.5 + 0.5)
                vis.img('rec_b', rec_b[0] * 0.5 + 0.5)
                vis.img('real_a', real_a[0] * 0.5 + 0.5)
                vis.img('real_b', real_b[0] * 0.5 + 0.5)
                vis.plot("lrschedu lr", optim_G.param_groups[0]['lr'])
                loss_dict_gan = {
                    'loss_G_A': float(loss_g_a),
                    'loss_G_B': float(loss_g_b),
                    'loss_D_A': float(loss_d_a),
                    'loss_D_B': float(loss_d_b),
                }
                vis.plot_many_in_one('GAN loss', loss_dict_gan)
                loss_dict_l1 = {
                    'loss_cycle_A': float(loss_cycle_a),
                    'loss_cycle_B': float(loss_cycle_b),
                }
                if opt.lambda_identity > 0:
                    loss_dict_l1['loss_idt_A'] = float(loss_idt_a)
                    loss_dict_l1['loss_idt_B'] = float(loss_idt_b)
                    vis.img('idt_a', idt_a[0] * 0.5 + 0.5)
                    vis.img('idt_b', idt_b[0] * 0.5 + 0.5)
                    vis.img('idt_aa', idt_aa[0] * 0.5 + 0.5)
                    vis.img('idt_bb', idt_bb[0] * 0.5 + 0.5)
                if opt.lambda_diff > 0:
                    loss_dict_l1['loss_diff_A'] = float(loss_diff_a)
                    loss_dict_l1['loss_diff_B'] = float(loss_diff_b)

                vis.plot_many_in_one('L1 loss', loss_dict_l1)
                fs.write('step {},loss: {}, {}\n'.format(step, loss_dict_gan, loss_dict_l1))
                fs.flush()

            if step % opt.test_freq == 0 and opt.need_test > 0:
                if opt.need_test == 1:
                    mean_loss = test(net_G_A, test_dataloader)
                    vis.plot_many(mean_loss)
                    fs.write('test: {}\n'.format(mean_loss))
                    print(mean_loss)
                    mean_loss = mean_loss['psnr']
                if mean_loss > best_psnr:
                    best_psnr = mean_loss
                    torch.save({
                        'net_G_A': net_G_A.state_dict(),
                        'net_G_B': net_G_B.state_dict(),
                        'net_G_AA': net_G_AA.state_dict(),
                        'net_G_BB': net_G_BB.state_dict(),
                        'net_D_A': net_D_A.state_dict(),
                        'net_D_B': net_D_B.state_dict(),
                        'optim_G': optim_G.state_dict(),
                        'optim_D': optim_D.state_dict(),
                        'sched_G': sched_G.state_dict(),
                        'sched_D': sched_D.state_dict(),
                        'step': step
                    }, os.path.join(opt.checkpoints_dir, opt.name, opt.name + '_best.pt'))

                    # print('sched_G=', sched_G.get_last_lr(), 'sched_D=', sched_D.get_last_lr())
                    print("save best checkpoint to", os.path.join(opt.checkpoints_dir, opt.name, opt.name + '_best.pt'))
                    fs.write("save best checkpoint to " + os.path.join(
                        opt.checkpoints_dir, opt.name, opt.name + '_best.pt') + '\n')
            if step % opt.test_freq == 0:
                torch.save({
                    'net_G_A': net_G_A.state_dict(),
                    'net_G_B': net_G_B.state_dict(),
                    'net_G_AA': net_G_AA.state_dict(),
                    'net_G_BB': net_G_BB.state_dict(),
                    'net_D_A': net_D_A.state_dict(),
                    'net_D_B': net_D_B.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                    'step': step
                }, os.path.join(opt.checkpoints_dir, opt.name, opt.name + '_last.pt'))
                print("save last checkpoint to", os.path.join(opt.checkpoints_dir, opt.name, opt.name + '_last.pt'))
                fs.write("save last checkpoint to " + os.path.join(
                    opt.checkpoints_dir, opt.name, opt.name + '_last.pt') + '\n')
            if step_now < opt.warmup_step:
                lr_warmup(optim_G, step_now, opt.warmup_step, 0, opt.lr_G)
                lr_warmup(optim_D, step_now, opt.warmup_step, 0, opt.lr_D)
                step_now += 1
            else:
                sched_G.step()
                sched_D.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ParamNet-aperio", type=str,
                        help="name of the experiment.")
    parser.add_argument("--train_dir_root",
                        default="/home/khtao/md3400-1/khtao_data/Dataset_Public/aperio_hamamatsu/train",
                        type=str,
                        help="path to images dir root for training")
    parser.add_argument("--test_dir_root",
                        default="/home/khtao/md3400-1/khtao_data/Dataset_Public/aperio_hamamatsu/test",
                        type=str,
                        help="path to images dir root for training")
    parser.add_argument("--dir_A", default="aperio", type=str, help="path to source images for training")
    parser.add_argument("--dir_B", default="hamamatsu", type=str, help="path to target images for training")
    parser.add_argument("--train_size", default=256, type=int, help="image size during training")
    parser.add_argument("--test_size", default=256, type=int, help="image size during testing")
    parser.add_argument("--total_steps", default=200000, type=int, help="total number of training steps")
    parser.add_argument('--batchsize', type=int, default=1, help='batch size')
    parser.add_argument("--lr_G", default=2e-4, type=float, help="Generator learning rate")
    parser.add_argument("--lr_D", default=2e-4, type=float, help="Discriminator learning rate")
    parser.add_argument("--betas", default=[0.5, 0.999], type=list, help="for Adam")

    parser.add_argument('--need_test', default=0, type=int, help='test generator during training')
    parser.add_argument("--random_scale", action='store_true', help="use random sacle train Discriminator")
    parser.add_argument("--pool_size", type=int, default=50, help="image pool size")
    parser.add_argument('--backbone', type=str, default='resnet18', help='the backbone of ParamNet')
    parser.add_argument('--resample_size', type=int, default=128, help='# of resample_size in ParamNet')
    parser.add_argument('--channels', type=int, default=8, help='# of channels in ParamNet')
    parser.add_argument('--pretrained', type=str,
                        default=None,
                        help='load pretrained paramnet')
    parser.add_argument('--step_count', type=int, default=1, help='step count')
    parser.add_argument('--continue_train', action='store_true', help='load last paramnet')
    parser.add_argument('--n_layer', type=int, default=2, help='# of layers in ParamNet')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.002,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--warmup_step', type=int, default=1000, help='learning rate warmup step')
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_diff', type=float, default=10.0, help='weight for diff paramnet and resnetgereator')
    parser.add_argument('--lambda_identity', type=float, default=2.0,
                        help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.'
                             ' For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss,'
                             ' please set lambda_identity = 0.1')
    parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--test_freq', type=int, default=5000, help='frequency of test Generator')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--device', type=str, default='0', help='run on # GPU')
    args = parser.parse_args()
    print_options(args, parser)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0')
    set_seed(args.seed)
    train(opt=args)

