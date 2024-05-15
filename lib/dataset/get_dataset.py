import torchvision.transforms as transforms
from lib.dataset.sewerml import SewerMLDataset
from lib.utils.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp

def get_datasets(args):
    datset_size = args.dataset_size
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                 RandAugment(),
                                 transforms.ToTensor(),
                                 normalize]
    try:
        # for q2l_infer scripts
        if args.cutout:
            print("Using Cutout!!!")
            train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    except Exception as e:
        Warning(e)
    train_data_transform = transforms.Compose(train_data_transform_list)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    if args.dataname == 'sewerml':
        dataset_dir = args.dataset_dir
        if datset_size == 1:
            train_dataset = SewerMLDataset(
                image_dir=osp.join(dataset_dir, 'complete_Sewer-ML/trainAll'),
                anno_path=osp.join(dataset_dir, 'complete_Sewer-ML/SewerML_Train.csv'),
                input_transform=train_data_transform,
            )
            val_dataset = SewerMLDataset(
                image_dir=osp.join(dataset_dir, 'complete_Sewer-ML/valAll'),
                anno_path=osp.join(dataset_dir, 'complete_Sewer-ML/SewerML_Val.csv'),
                # image_dir=osp.join(dataset_dir, 'Dataset_SewerML-Test'),
                # anno_path=osp.join(dataset_dir, 'Dataset_SewerML_sub_0.0625/testAll.csv'),
                input_transform=test_data_transform,
            )
        elif datset_size == 0.0625:

            train_dataset = SewerMLDataset(
                image_dir=osp.join(dataset_dir, 'Dataset_SewerML_sub_0.0625/trainAll_sub_0.0625'),
                anno_path=osp.join(dataset_dir, 'Dataset_SewerML_sub_0.0625/trainAll_sub_0.0625.csv'),
                input_transform=train_data_transform,
            )
            val_dataset = SewerMLDataset(
                image_dir=osp.join(dataset_dir, 'Dataset_SewerML_sub_0.0625/valAll_sub_0.0625'),
                anno_path=osp.join(dataset_dir, 'Dataset_SewerML_sub_0.0625/valAll_sub_0.0625.csv'),
                input_transform=test_data_transform,
            )
        else:
            raise NotImplementedError("Unknown dataset size %s" % args.dataset_size)

    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset