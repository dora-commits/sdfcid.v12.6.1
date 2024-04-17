import os
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.folder import is_image_file

from typing import Optional, Callable, List, Tuple, Dict, Any


class MVTecAD(VisionDataset):

    # urls from https://www.mvtec.com/company/research/datasets/mvtec-ad/
    data_dict = {
        'mvtec_anomaly_detection': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'
    }
    subset_dict = {
        'bottle' : 'https://drive.google.com/file/d/1lj4zKZt72HCVbcwqDGIY0O5__zMBfqKO/view?usp=drive_link',
    }

    # definition specified to MVTec-AD dataset
    dataset_name = next(iter(data_dict.keys()))
    subset_names = list(subset_dict.keys())
    normal_str = 'good'
    mask_str = 'ground_truth'
    train_str = 'train'
    test_str = 'test'
    compress_ext = '.tar.xz'
    image_size = (900, 900)

    def __init__(self,
                 root,
                 subset_name: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mask_transform: Optional[Callable] = None,
                 download=True,
                 pin_memory=False):

        super(MVTecAD, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train
        self.mask_transform = mask_transform
        self.download = download
        self.pin_memory = pin_memory

        # path
        self.dataset_root = os.path.join(self.root, self.dataset_name)
        self.subset_name = subset_name.lower()
        self.subset_root = os.path.join(self.dataset_root, self.subset_name)
        self.subset_split = os.path.join(self.subset_root, self.train_str if self.train else self.test_str)

        if self.download is True:
            self.download_subset()

        if not os.path.exists(self.subset_root):
            # print(self.subset_root)
            # print(self.subset_name)
            raise FileNotFoundError('subset {} is not found, please set download=True to download it.')

        # get image classes and corresponding targets
        self.classes, self.class_to_idx = self._find_classes(self.subset_split)

        # get image paths, mask paths and targets
        self.image_paths, self.mask_paths, self.targets = self._find_paths(self.subset_split, self.class_to_idx)
        if self.__len__() == 0:
            raise FileNotFoundError("found 0 files in {}\n".format(self.subset_split))

        # pin memory (usually used for small datasets)
        if self.pin_memory:
            self.data = self._load_images('RGB', self.image_paths)
            self.masks = self._load_images('L', self.mask_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        '''
        get item iter.
        :param idx (int): idx
        :return: (tuple): (image, mask, target) where target is index of the target class.
        '''
        # get image, mask and target of idx
        if self.pin_memory:
            image, mask = self.data[idx], self.masks[idx]
        else:
            image, mask = self._pil_loader('RGB', self.image_paths[idx]), self._pil_loader('RGB', self.mask_paths[idx])
        target = self.targets[idx]

        # apply transform
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.transform(mask)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, mask, target

    def __len__(self) -> int:
        return len(self.targets)

    def extra_repr(self):
        split = self.train_str if self.train else self.test_str
        return 'using data: {data}\nsplit: {split}'.format(data=self.subset_name, split=split)

    def download_subset(self):
        '''
        download the subset
        :return:
        '''
        os.makedirs(self.dataset_root, exist_ok=True)

        if os.path.exists(self.subset_root):
            return

        if self.subset_name not in self.subset_names:
            raise ValueError('The dataset called {} is not exist.'.format(self.subset_name))

        # download
        filename = self.subset_name + self.compress_ext
        download_and_extract_archive(self.subset_dict[self.subset_name], self.dataset_root, filename=filename)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.remove(self.normal_str)
        classes = [self.normal_str] + classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _find_paths(self,
                    folder: str,
                    class_to_idx: Dict[str, int]) -> Tuple[Any, Any, Any]:
        '''
        find image paths, mask paths and corresponding targets
        :param folder: folder/class_0/*.*
                       folder/class_1/*.*
        :param class_to_idx: dict of class name and corresponding label
        :return: image paths, mask paths, targets
        '''
        # define variables to fill
        image_paths, mask_paths, targets = [], [], []

        # define path find helper
        def find_mask_from_image(target_class, image_path):
            '''
            find mask path according to image path
            :param target_class: target class
            :param image_path: image path
            :return: None or mask path
            '''
            if target_class is self.normal_str:
                mask_path = None
            else:
                # only test data have mask images
                mask_path = image_path.replace(self.test_str, self.mask_str)
                fext = '.' + fname.split('.')[-1]
                mask_path = mask_path.replace(fext, '_mask' + fext)
            return mask_path

        # find
        for target_class in class_to_idx.keys():
            class_idx = class_to_idx[target_class]
            target_folder = os.path.join(folder, target_class)
            for root, _, fnames in sorted(os.walk(target_folder, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        # get image
                        image_paths.append(os.path.join(root, fname))
                        # get mask
                        mask_paths.append(find_mask_from_image(target_class, image_paths[-1]))
                        # get target
                        targets.append(class_idx)

        return image_paths, mask_paths, targets

    def _pil_loader(self, mode: str, path: str):
        '''
        load PIL image according to path.
        :param mode: PIL option, 'RGB' or 'L'
        :param path: image path, None refers to create a new image
        :return: PIL image
        '''
        if path is None:
            image = Image.new(mode, size=self.image_size)
        else:
            # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            # !!! directly using Image.open(mode, path) will lead to Dataloader Error inside loop !!!
            with open(path, 'rb') as f:
                image = Image.open(f)
                image = image.convert(mode)
        return image

    def _load_images(self, mode: str, paths: List[str]) -> List[Any]:
        '''
        load images according to paths.
        :param mode: PIL option, 'RGB' or 'L'
        :param paths: paths of images to load
        :return: list of images
        '''
        images = []
        for path in paths:
            images.append(self._pil_loader(mode, path))
        return images

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset, Subset


def _convert_label(x):
    '''
    convert anomaly label. 0: normal; 1: anomaly.
    :param x (int): class label
    :return: 0 or 1
    '''

    return 0 if x == 0 else 1

if __name__ == '__main__':

    # define transforms
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    target_transform = transforms.Lambda(_convert_label)

    # load data
    mvtec = MVTecAD('data',
                    subset_name='bottle',
                    train=True,
                    transform=transform,
                    mask_transform=transform,
                    target_transform=target_transform,
                    download=True)

    mvtec2 = MVTecAD('data',
                    subset_name='bottle',
                    train=False,
                    transform=transform,
                    mask_transform=transform,
                    target_transform=target_transform,
                    download=True)

    dataset = ConcatDataset([mvtec, mvtec2])

    # feed to data loader
    data_loader = DataLoader(dataset,
                             batch_size=dataset.__len__(),
                             shuffle=True,
                             num_workers=2,
                             pin_memory=False,
                             drop_last=True)

    # obtain in batch
    # for idx, (image, mask, target) in enumerate(data_loader):
    #     print(idx, target)

dec_x_f, _, dec_y_f = next(iter(data_loader))

FoldsX, FoldsY = cross_validation_5folds_MVTECAD(dec_x_f, dec_y_f)

dec_x_f = dec_x_f.reshape(dec_x_f.shape[0],3, 64, 64)

print('decy ',dec_y_f.shape)
print(collections.Counter(dec_y_f.numpy()))
print('train imgs after reshape ',dec_x_f.shape)
print(dec_x_f.shape[0])

classes = ['0', '1']
num_classes = len(classes)
samples_per_class = 5
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(dec_y_f == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        img=image.array_to_img((dec_x_f[idx].permute(1, 2, 0)), scale=True)
        plt.imshow(img)
        plt.axis('off')
        if i == 0:
            plt.title(cls)

plt.show()

combx_v = dec_x_f
comby_v = dec_y_f
     