import os
import random
import time
from glob import glob
import lmdb
from tqdm import tqdm
import numpy as np
import cv2
import torch
import visdom
from PIL import Image
from torch.utils.data import dataset


def list_file_tree(path, file_type="tif"):
    image_list = list()
    dir_list = os.listdir(path)
    if os.path.isdir(path):
        image_list += glob(os.path.join(path, "*" + file_type))
    for dir_name in dir_list:
        sub_path = os.path.join(path, dir_name)
        if os.path.isdir(sub_path):
            image_list += list_file_tree(sub_path, file_type)
    return image_list


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
class ImageDataset(dataset.Dataset):
    def __init__(self, dir_root, dir_A, dir_B, align=False, imgsize=256, transform=None):
        self.dir_root = dir_root
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.imgsize = imgsize
        self.source_list = []
        self.target_list = []
        self.transform = transform
        for dd in self.dir_A.split('+'):
            self.source_list += list_file_tree(os.path.join(self.dir_root, dd), "jpg")
            self.source_list += list_file_tree(os.path.join(self.dir_root, dd), "png")
            print(self.source_list[-1], len(self.source_list))
        for dd in self.dir_B.split('+'):
            self.target_list += list_file_tree(os.path.join(self.dir_root, dd), "jpg")
            self.target_list += list_file_tree(os.path.join(self.dir_root, dd), "png")
            print(self.target_list[-1], len(self.target_list))
        self.align = align
        if self.align:
            assert len(self.source_list) == len(self.target_list)
            self.target_list.sort()
            self.source_list.sort()
        else:
            random.shuffle(self.source_list)
            random.shuffle(self.target_list)
        self.A_size = len(self.source_list)  # get the size of dataset A
        self.B_size = len(self.target_list)  # get the size of dataset B

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, item):
        source_path = self.source_list[item % self.A_size]  # make sure index is within then range
        if self.align:  # make sure index is within then range
            target_path = self.target_list[item % self.B_size]
        else:  # randomize the index for target to avoid fixed pairs.
            item_t = random.randint(0, self.B_size - 1)
            target_path = self.target_list[item_t]
        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        if self.imgsize is not None:
            target_image = target_image.resize((self.imgsize, self.imgsize))
            source_image = source_image.resize((self.imgsize, self.imgsize))
        if self.transform:
            target_image = self.transform(target_image)
            source_image = self.transform(source_image)
        source_image = ((np.array(source_image, dtype=np.float32) - 127.5) / 127.5).transpose((2, 0, 1))
        target_image = ((np.array(target_image, dtype=np.float32) - 127.5) / 127.5).transpose((2, 0, 1))
        return source_image, target_image


class SingleImage(dataset.Dataset):
    def __init__(self, data_path, use_lmdb=True):
        self.data_path = data_path
        self.image_list = list_file_tree(data_path, "png")
        # self.image_list += list_file_tree(os.path.join(data_path), "jpg")
        self.use_lmdb = use_lmdb
        self.image_list.sort()
        if self.use_lmdb:
            self.lmdb_data = self.make_lmdb(os.path.join(self.data_path, 'lmdb'))

    def make_lmdb(self, path):
        length = len(self.image_list)
        if os.path.exists(path):
            env = lmdb.open(path, map_size=1099511627776)
            txn = env.begin()
            num = txn.get("len".encode())
            if num is None or int(txn.get("len".encode())) != length:
                os.remove(path + "/data.mdb")
                os.remove(path + "/lock.mdb")
            else:
                return txn
        env = lmdb.open(path, map_size=1099511627776)
        txn = env.begin(write=True)
        for idx in tqdm(range(length)):
            image = Image.open(self.image_list[idx]).convert("RGB")
            buff = cv2.imencode(".png", np.array(image, dtype=np.uint8))[1]
            txn.put(key=("image" + str(idx)).encode(), value=buff.tobytes())
        txn.put(key="len".encode(), value=str(length).encode())
        txn.commit()
        return env.begin()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        if self.use_lmdb:
            img = self.lmdb_data.get(key=("image" + str(item)).encode())
            img = np.frombuffer(img, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = Image.fromarray(img)
        else:
            img = Image.open(self.image_list[item]).convert("RGB").resize((256, 256))
        img = ((np.array(img, dtype=np.float32) - 127.5) / 127.5).transpose((2, 0, 1))
        return img


class Visualizer(object):
    """
    wrapper for visdom
    you can still access naive visdom function by
    self.line, self.scater,self._send,etc.
    due to the implementation of `__getattr__`
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self._vis_kw = kwargs

        # e.g.（’loss',23） the 23th value of loss
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        change the config of visdom
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def plot_many_in_one(self, display_name, losses):
        if not hasattr(self, display_name):
            setattr(self, display_name, {'X': [], 'Y': [], 'legend': list(losses.keys())})
        x = self.index.get(display_name, 0)
        plot_data = getattr(self, display_name)
        plot_data['X'].append(x)
        plot_data['Y'].append([losses[k] for k in plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
            Y=np.array(plot_data['Y']),
            opts={
                'title': display_name,
                'legend': plot_data['legend'],
                'xlabel': 'step',
                'ylabel': 'loss'},
            win=display_name)
        self.index[display_name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(
            env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
