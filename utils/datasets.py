#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


# utilities
# -----

# custom functions
# -----
def show_batch(sample_batched):
    """
    sample_batched: Tuple[torch.tensor, torch.tensor] -> None
    show_batch takes a contrastive sample sample_batched and plots
    an overview of the batch
    """

    grid_border_size = 2
    nrow = 10

    batch_1 = sample_batched[0][0][:, 0:, :, :]
    batch_2 = sample_batched[0][1][:, 0:, :, :]
    difference = np.abs(batch_1 - batch_2)

    titles = ["first contrast", "second contrast", "difference"]

    fig, axes = plt.subplots(1, 3, figsize=(2 * 6.4, 4.8))
    for (i, batch) in enumerate([batch_1, batch_2, difference]):
        ax = axes[i]
        grid = utils.make_grid(batch, nrow=nrow, padding=grid_border_size)
        ax.imshow(grid.numpy().transpose((1, 2, 0)))
        ax.set_title(titles[i])
        ax.axis("off")
    plt.show()


# ----------------
# custom classes
# ----------------

# custom CLTT dataset superclass (abstract)
# -----

class CLTTDataset(Dataset):
    """
    CLTTDataset is an abstract class implementing all the necessary methods
    to sample data according to the CLTT approach. CLTTDataset itself
    should not be instantiated as a standalone class, but should be
    inherited from and abstract methods should be overwritten
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
         n_fix=5, contrastive=True, sampling_mode='uniform', shuffle_object_order=True, circular_sampling=True, buffer_size=12096):
        """
        __init__ initializes the CLTTDataset Class, it defines class-wide
        constants and builds the registry of files and the data buffer
        
        root:str path to the dataset directory
        train:bool training set instead of testing
        transform:torchvision.transform
        target_transform:torchvision.transform
        n_fix:int for deterministic n_fix, float for probabilistic
        contrastive:bool contrastive dataset mode
        sampling_mode:str how the buffer gets built
        circular_sampling:bool make the first object the last object
        buffer_size:int approximate buffersize
        
        """
        super().__init__()
        
        self.train = train
        self.sampling_mode = sampling_mode
        self.shuffle_object_order = shuffle_object_order
        self.buffer_size = buffer_size
        self.n_fix = n_fix
        self.tau_plus = 1
        self.tau_minus = 0  # contrasts from the past (experimental)

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.contrastive = contrastive
        self.circular_sampling = circular_sampling

        self.get_dataset_properties()
        
        self.registry = self.build_registry(train)

        if self.contrastive:
            self.buffer = self.build_buffer(self.registry, self.sampling_mode, self.n_fix, self.shuffle_object_order, approx_size=self.buffer_size)
        else:
            # if used in non-contrastive mode the sampler just samples from all data
            self.buffer = self.registry
        pass

    def __len__(self):
        """
        __len__ defines the length of the dataset and indirectly
        defines how many samples can be drawn from the dataset
        in one epoch
        """
        length = len(self.buffer)
        return length
    
    def get_dataset_properties(self):
        """
        get_dataset_properties has to be defined for each dataset
        it stores number of objects, number of classes, a list of
        strings with labels
        """
        
        # basic properties (need to be there)
        self.n_objects = 3  # number of different objects >= n_classes
        self.n_classes = 3  # number of different classes
        self.labels = [
            "A",
            "B",
            "C",
            ]
        self.n_views_per_object = 10 # how many overall views of each object
        self.subdirectory = '/dataset_name/' # where is the dataset
        self.name = 'dataset name' # name of the dataset
        
        # custom properties (optional, dataset specific)
        # (anything you would want to have available in self)
        self.custom_property = ['one', 'two', 'three']
        
        raise Exception("Calling abstract method, please inherit \
        from the CLTTDataset class and reimplement this method") # pseudoclass 
        pass
    
        
    def __getitem__(self, idx):
        """
        __getitem__ is a method that defines how one sample of the
        dataset is drawn
        """
        if self.contrastive:
            image, label = self.get_single_item(idx)
            augmentation, _ = self.sample_contrast(idx)
        
            if self.transform:
                image, augmentation = self.transform(
                    image), self.transform(augmentation)
        
            if self.target_transform:
                label = self.target_transform(label)
        
            output = ([image, augmentation], label)
        else:
            image, label = self.get_single_item(idx)
        
            if self.transform:
                image = self.transform(image)
        
            if self.target_transform:
                label = self.target_transform(label)
        
            output = image, label
        
        return output
    
    def sample_contrast(self, chosen_index):
        """
        given index chosen_index, sample a corresponding contrast close in time
        """
        chosen_time = self.buffer.iloc[chosen_index]["time_idx"]
        
        possible_indices = self.buffer[
            (self.buffer["time_idx"].between(chosen_time - self.tau_minus, chosen_time + self.tau_plus)) & (
                    self.buffer["time_idx"] != chosen_time)].index
        
        # sampling at the end of the buffer
        if (chosen_time + self.tau_plus) > self.buffer.time_idx.max():
            if self.circular_sampling:
                also_possible = self.buffer[
                    (self.buffer["time_idx"].between(self.buffer.time_idx.min(), (
                            chosen_time + self.tau_plus - 1) - self.buffer.time_idx.max())) & (
                            self.buffer["time_idx"] != chosen_time)].index
            else:
                also_possible = self.buffer[self.buffer["time_idx"] == chosen_time].index
        
            possible_indices = possible_indices.union(also_possible)
        
        # sampling at the beginning of the buffer
        if (chosen_time - self.tau_minus) < self.buffer.time_idx.min():
            if self.circular_sampling:
                also_possible = self.buffer[
                    (self.buffer["time_idx"].between(self.buffer.time_idx.max() + (chosen_time - self.tau_minus) + 1,
                                                     self.buffer.time_idx.max())) & (
                            self.buffer["time_idx"] != chosen_time)].index
            else:
                also_possible = self.buffer[self.buffer["time_idx"] == chosen_time].index
        
            possible_indices = possible_indices.union(also_possible)
        
        chosen_index = np.random.choice(possible_indices)
        return self.get_single_item(chosen_index)

    
    def get_single_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, pd.core.indexes.numeric.Int64Index):
            idx = idx[0]
        
        path_to_file = self.buffer.loc[idx, "path_to_file"]
        if isinstance(path_to_file, pd.core.series.Series):
            path_to_file = path_to_file.item()
        
        image = Image.open(path_to_file)
        obj_info = self.buffer.iloc[idx, 1:].to_dict()
        
        label = self.buffer.loc[idx, "label"]
        return image, label
    
    def build_registry(self, train):
        """
        build a registry of all image files
        """
        path_list = []
        object_list = []
        label_list = []
        time_list = []
        
        d = self.root + self.subdirectory + 'train/' if train else self.root + self.subdirectory + 'test/'

        # have an ordered list
        list_of_files = os.listdir(d)
        list_of_files.sort()

        for timestep, path in enumerate(list_of_files):
            full_path = os.path.join(d, path)
            if os.path.isfile(full_path):
                path_list.append(full_path)
                object_list.append(timestep // self.n_views_per_object)
                label_list.append(timestep // self.n_views_per_object)
                time_list.append(timestep % self.n_views_per_object)
                
        tempdict = {'path_to_file': path_list, 'label': label_list, 'object_nr': object_list, 'time_idx': time_list}
        
        dataframe = pd.DataFrame(tempdict)
        dataframe.sort_values(by=['object_nr', 'time_idx'], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        
        
        return dataframe
    
    def build_buffer(self, registry, sampling_mode, n_fix, shuffle_object_order, approx_size):
        """
        build_buffer builds a buffer from all data that is available
        according to the sampling mode specified. Default method just
        returns the whole registry
        """
        
        # if n_fix is a probability, then get an expected value of the number of views
        expected_views = n_fix if n_fix >= 1 else self.expected_n(n_fix)
        
        object_order = np.arange(self.n_objects)
        if shuffle_object_order:
            np.random.shuffle(object_order)
    
        if sampling_mode == 'window':
            streambits = []
            for _ in range(approx_size // (round(expected_views) * self.n_objects)):
                for o in object_order:
                    n_views = self.get_n(n_fix) # get the n_fix for each object
                    chosen_index = np.random.choice(np.arange(0, self.n_views_per_object - n_views))
                    streambits.append(registry[registry.object_nr == o][
                                          registry.time_idx.between(chosen_index, chosen_index + n_views - 1)])
                    if shuffle_object_order:
                        np.random.shuffle(object_order)
    
            timestream = pd.concat(streambits, ignore_index=True)
            timestream.time_idx = np.arange(len(timestream.time_idx))
    
        elif sampling_mode == 'uniform':
            streambits = []
            for _ in range(approx_size // (round(expected_views) * self.n_objects)):
                for o in object_order:
                    n_views = self.get_n(n_fix) # get the n_fix for each object
                    chosen_indexs = np.random.choice(np.arange(0, self.n_views_per_object), n_views)
                    streambits.append(registry[registry.object_nr == o].iloc[chosen_indexs])
    
                    if shuffle_object_order:
                        np.random.shuffle(object_order)
    
            timestream = pd.concat(streambits, ignore_index=True)
            timestream.time_idx = np.arange(len(timestream.time_idx))
    
        elif sampling_mode == 'randomwalk':
            streambits = []
            for _ in range(approx_size // (round(expected_views) * self.n_objects)):
                for o in object_order:
                    n_views = self.get_n(n_fix) # get the n_fix for each object
                    streambits.append(registry.iloc[self.get_N_randomwalk_steps(n_views, o)])
    
            timestream = pd.concat(streambits, ignore_index=True)
            timestream.time_idx = np.arange(len(timestream.time_idx))
    
        else:
            print("[INFO] Warning, no sampling mode specified, defaulting to \
                whole dataset")
            timestream = registry #if no mode, then return the whole registry
    
        return timestream
    
    def refresh_buffer(self):
        """
        refresh buffer takes an CLTTDataset class and refreshes its own buffer
        given the registry
        """
        self.buffer = self.build_buffer(self.registry, self.sampling_mode, self.n_fix, self.shuffle_object_order, self.buffer_size)
        pass

    def get_N_randomwalk_steps(self, N, object_nr):
        """
        Get index values of N random walk steps of a object specified by "object_nr".
        """
        raise Exception("Calling abstract method, please inherit \
        from the CLTTDataset class and reimplement this method") # pseudoclass
        pass
    
    def expected_n(self, probability):
        """
        expected_n takes a float probability between 0 and 1
        and returns the expected value of the number of fixations
        """
        result = (1-probability)*(probability)/(1-(probability))**2 + 1
        return result
    
    def get_n(self, input):
        """
        get_n takes a float probability input between 0 and 1
        and returns n fixations according to probability
        if input >= 1 it just returns its argument
        """
        if input >= 1:
            return input
        else:
            result = 1 # make sure that you switch to the next object once
            while input > np.random.random():
                result += 1
            return result



# datasets (CLTTDataset subclasses)
# -----

# TODO: Rewrite MiyashitaDataset to be compatible with probabilistic n_fix

class MiyashitaDataset(CLTTDataset):
    """
    MiyashitaDataset is a dataset inspired by the works of
    Miyashita, 1988 it is comprised of a set of different fractal patterns that
    are presented in a specific order to be associated.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    
    def get_dataset_properties(self):
        
        # basic properties (need to be there)
        self.n_objects = 100  # number of different objects >= n_classes
        self.n_classes = 100  # number of different classes
        self.labels = [str(i) for i in range(self.n_classes)]

        self.n_views_per_object = 100 if (self.train and self.contrastive) else 1
        #self.n_fix # how many overall views of each object
        self.subdirectory = '/fractals100_64x64/' # where is the dataset
        self.name = 'Miyashita Fractals' # name of the dataset


        # custom properties (optional, dataset specific)
        # (anything you would want to have available in self)
        
        # for Miyashita every mode is the same
        # that means we do not need to reimplement get_n_randomwalk_steps
        # and can just fix the sampling mode
        self.sampling_mode = "uniform" if (self.train and self.contrastive) else "" # overwrite sampling mode
        
        self.basic_transform = transforms.RandomAffine(
            degrees=(-10, 10),
            translate=(0.15, 0.15),
            scale=(0.9, 1.0))
        
        # add the basic transform the the regular transform for training
        if (self.train and self.contrastive):
            self.transform = transforms.Compose([
                self.basic_transform,
                self.transform,
            ])
        
        pass
    
    def build_registry(self, train):
        """
        Reimplementation of the build_registry method, because Miyashita
        Fractals have no testset and the in-class variability is generated
        virtually instead of having multiple pictures
        """
        path_list = []
        object_list = []
        label_list = []
        time_list = []
        e = 0

        d = self.root + self.subdirectory # there is no fractals testset

        # have an ordered list
        list_of_files = os.listdir(d)
        list_of_files.sort()
        for o, path in enumerate(list_of_files):
            full_path = os.path.join(d, path)
            if os.path.isfile(full_path):
                repetitions = self.n_views_per_object
                # repeat the same picture n_fix times
                for timestep in range(repetitions):
                    path_list.append(full_path)
                    time_list.append(timestep + e * self.n_views_per_object)
                    object_list.append(o)
                    label_list.append(o)
                e += 1

        temporary_dict = {'path_to_file': path_list,
                          'label': label_list,
                          'object_nr': object_list,
                          'time_idx': time_list}

        dataframe = pd.DataFrame(temporary_dict)
        dataframe.sort_values(by=['object_nr', 'time_idx'], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)

        return dataframe

    

class TDWDataset(CLTTDataset):
    """
    The ThreeDWorld Dataset by Felix Schneider is
    comprised of 1008 views around 12 distinct objects rendered
    in the TDW environment
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    
    def get_dataset_properties(self):
        
        # basic properties (need to be there)
        self.n_objects = 12  # number of different objects >= n_classes
        self.n_classes = 12  # number of different classes
        self.labels = [
            "cup",
            "comb",
            "scissor",
            "hammer",
            "book",
            "calculator",
            "goblet",
            "candle",
            "headphones",
            "screwdriver",
            "cassette",
            "bottle",
        ]
        
        delta_phi = 10
        self.phis = np.arange(0, 360, delta_phi)
        delta_theta = 10
        self.thetas = np.arange(10, 80, delta_theta)
        delta_r = 0.1
        self.rs = np.arange(0.3, 0.7, delta_r)

        self.n_views_per_object = len(self.phis) * len(self.thetas) * len(self.rs) # how many overall views of each object

        self.subdirectory = '/spherical_photoreal_64x64_DoF/' # where is the dataset
        self.name = 'ThreeDWorld Objects' # name of the dataset


        # custom properties (optional, dataset specific)
        # (anything you would want to have available in self)
        pass
    
    def get_N_randomwalk_steps(self, N, object_nr):
        """
        Get index values of N random walk steps of a object specified by "object_nr".
        """
        
        def get_registry_index(r, theta, phi):
            """
            helper function to get index given a coordinate tuple,
            i.e. r, theta and phi value
            """
            ix = r * (len(self.thetas) * len(self.phis)) + theta * len(self.phis) + phi
            return ix
        
        index = []

        # Possible values for r,theta and phi
        r = len(self.rs)
        theta = len(self.thetas)
        phi = len(self.phis)

        # select random start values for r,theta and phi
        current_r = np.random.randint(0, r - 1)
        current_theta = np.random.randint(0, theta - 1)
        current_phi = np.random.randint(0, phi - 1)

        for i in range(N):
            while True:
                # 6 possible direction in which to go from the current position
                # Possible steps: +/-r, +/-Phi, +/-Theta
                rand = np.random.randint(low=0, high=5)

                # For the choosen step direction, it will be checked if this is a "valid".
                if (rand == 0) & (current_r < r - 1):
                    current_r += 1
                    break
                if (rand == 1) & (current_r > 0):
                    current_r -= 1
                    break
                if (rand == 2) & (current_theta < theta - 1):
                    current_theta += 1
                    break
                if (rand == 3) & (current_theta > 0):
                    current_theta -= 1
                    break
                if (rand == 4) & (current_phi < phi - 1):
                    current_phi += 1
                    break
                if (rand == 5) & (current_phi > 0):
                    current_phi -= 1
                    break

            # transform r,theta, phi values 
            # into index number between 0 and 1008
            ix = get_registry_index(
                    current_r, current_theta, current_phi)
            index.append(ix)
        index = np.array(index)

        # to get index values for object "object_nr", the values are shifted
        index += self.n_views_per_object * object_nr
        return index
    
    def additional_metadata(self):
        # hacky way to get some metadata, to be revised
        phi_angle_list = []
        theta_angle_list = []
        radius_list = []
        for o in range(self.n_classes):
            for r in self.rs:
                for theta in self.thetas:
                    for phi in self.phis:
                        phi_angle_list.append(phi)
                        theta_angle_list.append(theta)
                        radius_list.append(r)
    
        tempdict = {'phi': phi_angle_list, 'theta': theta_angle_list, 'radius': radius_list}
        dataframe = pd.DataFrame(tempdict)
        self.registry= pd.merge(self.registry, dataframe, left_index=True, right_index=True)
        pass



class COIL100Dataset(CLTTDataset):
    """
    COIL100Dataset is a dataset by the work of Sameer, Shree, and Hiroshi, 1996.
    It is comprised of color images of 100 objects, and each object has 72 views.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def get_dataset_properties(self):

        # basic properties (need to be there)
        self.n_objects = 100  # number of different objects >= n_classes
        self.n_classes = 100  # number of different classes
        self.labels = [str(i) for i in range(self.n_classes)]
    
        if self.train:
            self.n_views_per_object = 54   # number of overall views of each object on trainset
        else:
            self.n_views_per_object = 18   # number of overall views of each object on testset
    
        self.subdirectory = '/coil100_128x128/'  # where is the dataset
        self.name = 'Columbia University Image Library'  # name of the dataset

        # custom properties (optional, dataset specific)
        # (anything you would want to have available in self)

        pass

    def get_N_randomwalk_steps(self, N, object_nr):
        """
        Get index values of N random walk steps of a object specified by "object_nr".
        """
        index = []
        current_idx = np.random.randint(0, self.n_views_per_object - 1)
        for i in range(N):
            while True:
                # 2 possible direction in which to go from the current position
                # Possible steps: +: left, -: right
                rand = np.random.randint(low=0, high=2)
                if (rand == 0) & (current_idx > 0):
                    current_idx -= 1
                    break
                if (rand == 1) & (current_idx < self.n_views_per_object - 1):
                    current_idx += 1
                    break
            index.append(current_idx)
        index = np.array(index)
        index += self.n_views_per_object * object_nr
        return index

    def build_registry(self, train):
        """
        build a registry of all image files
        """
        path_list = []
        object_list = []
        label_list = []
        time_list = []

        # d = self.root + self.subdirectory
        d = self.root + self.subdirectory + 'train/' if train else self.root + self.subdirectory + 'test/'


        # have an ordered list
        list_of_files = os.listdir(d)
        list_of_files.sort()

        for timestep, path in enumerate(list_of_files):
            full_path = os.path.join(d, path)
            if os.path.isfile(full_path):
                path_list.append(full_path)
                object_list.append(timestep // self.n_views_per_object)
                label_list.append(timestep // self.n_views_per_object)
                time_list.append(timestep % self.n_views_per_object)

        tempdict = {'path_to_file': path_list, 'label': label_list, 'object_nr': object_list, 'time_idx': time_list}

        dataframe = pd.DataFrame(tempdict)
        dataframe.sort_values(by=['object_nr', 'time_idx'], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe


# ----------------
# main program
# ----------------

if __name__ == "__main__":

    # Miyashita Dataset
    # -----

    dataset = MiyashitaDataset(
        root='../data',
        train=True,
        transform=transforms.ToTensor(),
        contrastive=True,
        shuffle_object_order=False,
        circular_sampling=True,
        buffer_size = 500,
    )

    # original timeseries
    dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)
    for ibatch, sample_batched in enumerate(dataloader):
        show_batch(sample_batched)

    # shuffled timeseries
    dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=True)
    for ibatch, sample_batched in enumerate(dataloader):
        show_batch(sample_batched)

    # TDW Dataset
    # -----

    dataset = TDWDataset(
        root='../data',
        train=True,
        transform=transforms.ToTensor(),
        n_fix = 5,
        contrastive=True,
        sampling_mode='randomwalk',
    )

    # original timeseries
    dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)
    for ibatch, sample_batched in enumerate(dataloader):
        show_batch(sample_batched)
        if ibatch == 3:
            break

    # shuffled timeseries
    dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=True)
    for ibatch, sample_batched in enumerate(dataloader):
        show_batch(sample_batched)
        if ibatch == 3:
            break
    
    # this is how you can implement refreshing the buffer
    # ideally this should happen at the dataloader level, but for now it is defined
    # within the dataset

    dataset = TDWDataset(
        root='../data',
        train=True,
        transform=transforms.ToTensor(),
        contrastive=True,
        sampling_mode='randomwalk',
        buffer_size=1000,
        n_fix=.5)

    dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)
    epochs = 5
    for e in range(epochs):
        for ibatch, sample_batched in enumerate(dataloader):
            print(sample_batched[0][0].shape)
            if ibatch == 0:
                show_batch(sample_batched)
        # end of epoch
        # rebuild the buffer
        dataset.refresh_buffer()

    # COIL100 Dataset
    # -----

    dataset = COIL100Dataset(
        root='../data',
        train=True,
        transform=transforms.ToTensor(),
        n_fix=5,
        contrastive=True,
        sampling_mode='randomwalk',
    )

    # original timeseries
    dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)
    for ibatch, sample_batched in enumerate(dataloader):
        show_batch(sample_batched)

    # shuffled timeseries
    dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=True)
    for ibatch, sample_batched in enumerate(dataloader):
        show_batch(sample_batched)


# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
