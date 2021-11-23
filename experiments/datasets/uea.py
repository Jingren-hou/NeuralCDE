import collections as co
import numpy as np
import os
import pathlib
import sktime.utils.load_data
import torch
import urllib.request
import zipfile

from . import common


here = pathlib.Path(__file__).resolve().parent


def download():
    base_base_loc = here / 'data'
    base_loc = base_base_loc / 'UEA'
    loc = base_loc / 'Multivariate2018_ts.zip'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_base_loc):
        os.mkdir(base_base_loc)
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    # 国内网络下载实在有毛病，所以手动下载了数据集放在loc里，这里注释掉防止又自动下载报错。该urlretrieve命令可以下载并解压文件。
    # urllib.request.urlretrieve('http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip',
    #                           str(loc))

    with zipfile.ZipFile(loc, 'r') as f:
        f.extractall(str(base_loc))


# Is this actually necessary?
def _pad(channel, maxlen):
    # padding channel using the last value to maxlen
    channel = torch.tensor(channel)
    out = torch.full((maxlen,), channel[-1])
    out[:channel.size(0)] = channel
    return out

# 可用的替代UEA数据集，详情见’The UEA multivariate time series classification archive, 2018‘dataset介绍
valid_dataset_names = {'ArticularyWordRecognition',
                       'FaceDetection',
                       'NATOPS',
                       'AtrialFibrillation',
                       'FingerMovements',
                       'PEMS - SF',
                       'BasicMotions',
                       'HandMovementDirection',
                       'PenDigits',
                       'CharacterTrajectories',
                       'Handwriting',
                       'PhonemeSpectra',
                       'Cricket',
                       'Heartbeat',
                       'RacketSports',
                       'DuckDuckGeese',
                       'InsectWingbeat',
                       'SelfRegulationSCP1',
                       'EigenWorms',
                       'JapaneseVowels',
                       'SelfRegulationSCP2',
                       'Epilepsy',
                       'Libras',
                       'SpokenArabicDigits',
                       'ERing',
                       'LSST',
                       'StandWalkJump',
                       'EthanolConcentration',
                       'MotorImagery',
                       'UWaveGestureLibrary'}


def _process_data(dataset_name, missing_rate, intensity):
    # We begin by loading both the train and test data and using our own train/val/test split.
    # The reason for this is that (a) by default there is no val split and (b) the sizes of the train/test splits are
    # really janky by default. (e.g. LSST has 2459 training samples and 2466 test samples.)

    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    base_filename = here / 'data' / 'UEA' / 'Multivariate_ts' / dataset_name / dataset_name
    train_X, train_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TRAIN.ts')
    test_X, test_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TEST.ts')
    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()
    # for"CharacterTrajectories"data, train_X.shape = (1422, 3), train_y.shape =(1422), test_X = (1436, 3), test_y = (1436)
    X = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)
    # X.shape = (2858, 3), 其中3是channel, 每个channel是一个numpy的series类型, 储存着一定长度的单变量序列eg:
    # {Series:(116,)}(0, -0.13015)(1, -0.1813121)(2, -0.0234104)...(115, 0.67359)
    # y.shape = (2858)，type=str

    # 每个batch数据的时间长度可能不同，这里记录下来了lengths: tensor([116, 131, 140, ..., 150, 160, 143])
    lengths = torch.tensor([len(Xi[0]) for Xi in X])
    final_index = lengths - 1
    maxlen = lengths.max()      # 对于"CharacterTrajectories"data，maxlen=182
    # X is now a numpy array of shape (batch, channel)
    # Each channel is a pandas.core.series.Series object of length corresponding to the length of the time series
    X = torch.stack([torch.stack([_pad(channel, maxlen) for channel in batch], dim=0) for batch in X], dim=0)
    # X is a tensor of shape (batch, channel, length(padding to maxlen)) now
    X = X.transpose(-1, -2)
    # X is now a tensor of shape (batch, length, channel)
    times = torch.linspace(0, X.size(1) - 1, X.size(1))
    # times = tensor([0, 1, 2, ..., length-1])

    generator = torch.Generator().manual_seed(56789)
    for Xi in X:
        removed_points = torch.randperm(X.size(1), generator=generator)[:int(X.size(1) * missing_rate)].sort().values
        Xi[removed_points] = float('nan')

    # Now fix the labels to be integers from 0 upwards
    targets = co.OrderedDict()      # 有序字典，建立y中的str类别到数字标签的映射
    counter = 0
    for yi in y:
        if yi not in targets:
            targets[yi] = counter
            counter += 1
    # target=OrderedDict([('1',0),('2',1),('3',2),...,('20',19])

    y = torch.tensor([targets[yi] for yi in y])     # 将y中的str类别映射成数字标签

    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, input_channels) = common.preprocess_data(times, X, y, final_index, append_times=True,
                                                                append_intensity=intensity)

    num_classes = counter

    assert num_classes >= 2, "Have only {} classes.".format(num_classes)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, num_classes, input_channels)


def get_data(dataset_name, missing_rate, device, intensity, batch_size):
    # We begin by loading both the train and test data and using our own train/val/test split.
    # The reason for this is that (a) by default there is no val split and (b) the sizes of the train/test splits are
    # really janky by default. (e.g. LSST has 2459 training samples and 2466 test samples.)

    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    base_base_loc = here / 'processed_data'
    base_loc = base_base_loc / 'UEA'
    loc = base_loc / (dataset_name + str(int(missing_rate * 100)) + ('_intensity' if intensity else ''))
    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
        num_classes = int(tensors['num_classes'])
        input_channels = int(tensors['input_channels'])
    else:
        download()
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index, num_classes, input_channels) = _process_data(dataset_name, missing_rate, intensity)
        common.save_data(loc,
                         times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index,
                         num_classes=torch.as_tensor(num_classes), input_channels=torch.as_tensor(input_channels))

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, device,
                                                                                num_workers=0, batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels
