import pylidc as pl
import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm


# Paths to read from/write to
img_path = '/vol/vipdata/data/chest_ct/LIDC-IDRI/LIDC-IDRI/'
save_path = '/vol/vipdata/data/chest_ct/LIDC-IDRI/nodules_only'

img_prefix = 'LIDC-IDRI-'


def centre_to_slice(centre, pad):
    slices = []
    for i, c in enumerate(centre):
        slices += [slice(c - pad[i], c + pad[i], None)]

    return slices


def change_spacing(img, spc=np.array([1, 1, 1])):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator = sitk.sitkLinear
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputSpacing(spc.tolist())
    orig_size = np.array(img.GetSize(), dtype=np.int)
    orig_spacing = np.array(img.GetSpacing())
    new_size = orig_size * (orig_spacing / spc)
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    return resample.Execute(img)


def slice_vol(vol, ann_slice):
    dims = vol.shape
    pad = np.zeros((3, 2))
    new_slice = np.array(ann_slice)
    for i in range(len(new_slice)):
        if new_slice[i].start < 0:
            pad[i][0] = abs(new_slice[i].start)
            start = 0
        else:
            start = new_slice[i].start
        if new_slice[i].stop > dims[i]:
            pad[i][1] = new_slice[i].stop - dims[i]
            stop = dims[i]
        else:
            stop = new_slice[i].stop

        new_slice[i] = slice(start, stop, None)

    pad = tuple([tuple([int(tt) for tt in t]) for t in pad])

    return np.pad(vol[tuple(new_slice)], pad_width=pad, constant_values=0)


subject_list = os.listdir(img_path)
subject_list = [s for s in subject_list if s.startswith(img_prefix)]
for pid in tqdm(subject_list):
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid)
    for i_img in range(scans.count()):
        scan = scans[i_img]
        anns = scan.cluster_annotations()
        pad = 45 * np.ones(3)
        spacing = np.array([scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing])
        for i_nod, ann in enumerate(anns):
            l = len(ann)
            if l <= 4:
                centre = [0, 0, 0]
                for a in ann:
                    centre += a.centroid / l
                ann_slice = centre_to_slice([int(c) for c in centre], (pad / np.array([scan.pixel_spacing, scan.pixel_spacing, scan.slice_thickness])).astype(int))
                vol = scan.to_volume()
                # nod = vol[tuple(ann_slice)]
                nod = slice_vol(vol, ann_slice)
                img = sitk.GetImageFromArray(nod)
                img.SetSpacing(spacing)
                img = change_spacing(img)
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(save_path,
                                                scan.patient_id + '_im' + str(i_img) + '_nod' + str(i_nod) + '.nii.gz'))
                writer.Execute(img)

                for ann_i, a in enumerate(ann):
                    mask = a.boolean_mask()
                    box = a.bbox()
                    target = np.zeros(vol.shape)
                    target[box] = mask
                    # target = target[tuple(ann_slice)]
                    target = slice_vol(target, ann_slice)

                    img = sitk.GetImageFromArray(target)
                    img.SetSpacing(spacing)
                    img = change_spacing(img)
                    writer = sitk.ImageFileWriter()
                    writer.SetFileName(os.path.join(save_path,
                                                    scan.patient_id + '_im' + str(i_img) + '_nod' + str(i_nod) +
                                                    '_ann' + str(ann_i) + '.nii.gz'))
                    writer.Execute(img)

            if l < 4:
                blank_image = np.zeros((90, 90, 90))
                img = sitk.GetImageFromArray(blank_image)
                img.SetSpacing((1, 1, 1))
                for ann_i in range(l, 4):
                    writer = sitk.ImageFileWriter()
                    writer.SetFileName(os.path.join(save_path,
                                                    scan.patient_id + '_im' + str(i_img) + '_nod' + str(i_nod) +
                                                    '_ann' + str(ann_i) + '.nii.gz'))
                    writer.Execute(img)
