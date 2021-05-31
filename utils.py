import os
import time
import logging
import uuid
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
import shutil
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

"""
Logging utils
"""

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


def accuracy_search(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def save(model, model_path):
  torch.save(model.state_dict(), model_path)


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    # import ipdb; ipdb.set_trace()
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if 'graph_bn' not in name and 'GCN' not in name and 'transformation' not in name) / 1e6


def init_logger(logpath, experiment_name="sample", filepath=None, package_files=None, view_excuted_file=False,
                displaying=True, saving=True, debug=False, tqdm=True):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)
    st = time.gmtime()
    experiment_name = experiment_name + "-" + "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(
        st.tm_year, st.tm_mon, st.tm_mday, st.tm_hour, st.tm_min, st.tm_sec)

    if saving:
        info_file_handler = logging.FileHandler(
            os.path.join(logpath, experiment_name), mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if tqdm:
        tqdm_handler = TqdmLoggingHandler(level=logging.INFO)
        logger.addHandler(tqdm_handler)

    # Time
    start_time = time.strftime("%Y-%m-%d")
    excution_id = str(uuid.uuid4())[:8]
    logger.info("Experiment name : {}".format(experiment_name))
    logger.info("Start tiem : {}".format(start_time))
    logger.info("Execution ID : {}".format(excution_id))

    # For viewing whole codes
    if view_excuted_file:
        logger.info("=" * 80)
        logger.info("excuted file : {}".format(filepath))
        logger.info("=" * 80)
        with open(filepath, "r") as f:
            logger.info(f.read())

        for f in package_files:
            logger.info("package files : {}".format(f))
            with open(f, "r") as package_f:
                logger.info(package_f.read())

    return logger


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


"""
Method utils
"""


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, device, repeat=1, save_adv=None, criterion=None, attack=None, binarize=False):
    if save_adv is not None:
        writer = SummaryWriter(log_dir=save_adv)
    total_correct = 0
    criterion = criterion or torch.nn.CrossEntropyLoss().to(device)
    #max_x, min_x = -100., 100.
    for i, (x, y) in enumerate(dataset_loader):
        if attack is not None:
            x_nat = x.detach().clone()
            x = attack.perturb(x.to(device), y.to(device), device=device)
            if repeat != 1:
                y = torch.cat([y for _ in range(repeat)])
            if save_adv is not None:
                nat_image = torchvision.utils.make_grid(
                    x_nat.cpu(), scale_each=False)
                adv_image = torchvision.utils.make_grid(
                    x.cpu(), scale_each=False)
                writer.add_image("natural_image", nat_image, i)
                writer.add_image("adversarial_image", adv_image, i)

        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y).cpu().detach().numpy()
        predicted_class = torch.argmax(pred.cpu().detach(), dim=1)
        correct = (predicted_class == y.cpu())
        total_correct += torch.sum(correct).item()

    if save_adv is not None:
        writer.close()
    return total_correct / (len(dataset_loader.dataset) * repeat), loss


"""
Visualize utils
"""


def converter(image):
    convert = transforms.ToPILImage()
    image = torch.tensor(image)
    image = image.resize(image.size(-3), image.size(-2), image.size(-1))
    return convert(image)


def subset_sampler(source, num_image):
    # source : torchvision.datasets format
    subset_indice = list(torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(source), batch_size=num_image, drop_last=True))[0]
    subset = torch.utils.data.Subset(source, subset_indice)
    return subset


class RunningAverageMeter(object):
    """Computes and stores the averate and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == "__main__":
    init_logger("logs")
