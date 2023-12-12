import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from .randaugment import RandomAugment
import copy
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

'''
    :args: batch_size,dataset,data_dir,partial_rate,imb_type,imb_ratio,seed,data_dir_prod
'''
def load_cub200(args):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    batch_size = args.batch_size
    input_size = 224

    test_transform = transforms.Compose(
        [
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_dataset = Cub2011(root=args.data_dir, train=False, transform=test_transform, download=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size * 4, shuffle=False,
                                              num_workers=8)

    print('==> Loading local data copy in the long-tailed setup')
    data_file = "{ds}_{pr}_imb_{it}{imf}_sd{sd}.npy".format(
        ds=args.dataset,
        pr=args.partial_rate,
        it=args.imb_type,
        imf=args.imb_ratio,
        sd=args.seed)

    save_path = os.path.join(args.data_dir_prod, data_file)
    if not os.path.exists(save_path):
        data_dict = generate_data(data_dir=args.data_dir, save_path=save_path, partial_rate=args.partial_rate, imb_type=args.imb_type, imb_ratio=args.imb_ratio)
    else:
        data_dict = np.load(save_path, allow_pickle=True).item()


    train_filenames, train_labels = data_dict['train_filenames'], data_dict['train_labels']
    train_labels = torch.from_numpy(train_labels)
    partialY = torch.from_numpy(data_dict['partial_labels'])

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), train_labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')
    # check partial labels
    print('Average candidate num: ', partialY.sum(1).mean())

    train_label_cnt = torch.unique(train_labels, sorted=True, return_counts=True)[-1]
    # train_label_cnt is also used for intialize Acc-shot object

    base_folder = os.path.join(args.data_dir, Cub2011.base_folder)
    partial_matrix_dataset = CUB200_Augmentention(base_folder, train_filenames, partialY.float(), train_labels.float())
    # generate partial label dataset
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset,
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              num_workers=8,
                                                              pin_memory=True,
                                                              drop_last=True)

    # estimation loader for distribution estimation
    est_dataset = CUB200_Augmentention(base_folder, train_filenames, partialY.float(), train_labels, test_transform)
    est_loader = torch.utils.data.DataLoader(dataset=est_dataset,
                                             batch_size=batch_size * 4,
                                             shuffle=False,
                                             num_workers=8,
                                             pin_memory=True)

    return partial_matrix_train_loader, partialY, test_loader, est_loader, train_label_cnt


class CUB200_Augmentention(Dataset):
    '''
        base_folder: The base folder of images
        filenames: The filename of images
        given_label_matrix: partial label matrix
        true_labels: ground truth label
        transform: torchvision.transforms
    '''
    def __init__(self, base_folder, filenames, given_label_matrix, true_labels, transform=None):
        self.filenames = filenames
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
        self.transform = transform
        self.input_size = 224
        self.base_folder = base_folder
        self.loader = default_loader

        if self.transform is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            self.weak_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            self.strong_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        path = os.path.join(self.base_folder, self.filenames[index])
        # target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is None:
            each_image_w = self.weak_transform(img)
            each_image_s = self.strong_transform(img)
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]

            return each_image_w, each_image_s, each_label, each_true_label, index
        else:
            each_label = self.given_label_matrix[index]
            each_image = self.transform(img)
            each_true_label = self.true_labels[index]
            return each_image, each_label, each_true_label


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        # self.get_all = get_all_data

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')



    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        _data = images.merge(image_class_labels, on='img_id')
        self._data = _data.merge(train_test_split, on='img_id')

        if self.train:
            self._data = self._data[self._data.is_training_img == 1]
        else:
            self._data = self._data[self._data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self._data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sample = self._data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_data(self):
        return self._data


def generate_data(data_dir, save_path, partial_rate, imb_type, imb_ratio):
    # First, the long-tailed learning is known to be unstable,
    # we recommend running Our code with a pre-processed data copy,
    # which can be used for other baseline models as well.
    print('''This is the first time you run this setup.
        Generating local data copies ...''')

    train_filenames, train_labels, partial_labels = generate_imb_pll_data(data_dir, partial_rate, imb_type, imb_ratio)
    data_dict = {
            'train_filenames': train_filenames,
            'train_labels': train_labels.numpy(),
            'partial_labels': partial_labels.numpy()
        }

    with open(save_path, 'wb') as f:
        np.save(f, data_dict)
    print('local data saved at ', save_path)
    return data_dict

def generate_imb_pll_data(data_dir, partial_rate, imb_type, imb_ratio):
    cub200 = Cub2011(root=data_dir, train=True, transform=None, download=True)
    data = cub200.get_data().values
    _, filenames, train_labels, _ = np.split(data, 4, axis=1)
    filenames, train_labels = np.squeeze(filenames), np.squeeze(train_labels)
    num_class = np.max(train_labels)
    train_labels = train_labels - 1
    train_filenames, train_labels = gen_imbalanced_data(filenames, train_labels, num_class, imb_type, imb_ratio)
    partialY = generate_uniform_cv_candidate_labels(train_labels, partial_rate)

    return train_filenames, train_labels, partialY


def gen_imbalanced_data(filenames, targets, num_class, imb_type, imb_ratio):
    img_max = len(filenames) / num_class
    img_num_per_cls = get_img_num_per_cls(num_class, imb_type, 1/imb_ratio, img_max)

    new_filenames = []
    new_targets = []
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)
    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_filenames.append(filenames[selec_idx, ...])

        new_targets.extend([the_class, ] * the_img_num)

    new_filenames = np.concatenate(new_filenames)
    new_targets = torch.Tensor(new_targets).long()

    return new_filenames, new_targets


def get_img_num_per_cls(cls_num, imb_type, imb_factor, img_max):
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        raise NotImplementedError("You have chosen an unsupported imb type.")
    return img_num_per_cls


def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))]=p_1
    print('==> Transition Matrix:')
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY
