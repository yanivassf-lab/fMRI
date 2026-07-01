from nilearn.masking import compute_epi_mask
from nilearn import image

from nilearn.masking import compute_epi_mask
from nilearn import image

# git clone https://github.com/OpenNeuroDatasets/ds000113.git
# cd ds000113/sub-01/ses-movie/func
# datalad get datalad get sub-01_ses-movie_task-movie_run-2_bold.nii.gz

fmri_img = image.load_img('../../../../input_files/movie_files/sub-01_ses-movie_task-movie_run-2_bold.nii.gz')
mask_img = compute_epi_mask(fmri_img)
mask_img.to_filename('../../../../input_files/movie_files/sub-01_ses-movie_task-movie_run-2_brainmask.nii.gz')
