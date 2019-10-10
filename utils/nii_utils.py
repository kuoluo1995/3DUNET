import nibabel
from pathlib import Path


def nii_reader(path):
    image = nibabel.load(str(path))
    image_array = image.get_fdata()
    return image_array


def nii_header_reader(path):
    image = nibabel.load(str(path))
    image_header = image.header
    pix_dim = image_header.get('pixdim')
    image_affine = image.affine
    return {'header': image_header, 'affine': image_affine, 'spacing': (pix_dim[1], pix_dim[2], pix_dim[3])}


def nii_writer(path, header, image_array):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image = nibabel.Nifti1Image(image_array, affine=header['affine'], header=header['header'])
    nibabel.save(image, str(path))
