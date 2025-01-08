import os 
import numpy as np
import nibabel as nib 

# paths = ["/fs5/p_masi/krishar1/MIDL/B30f_B50f/hard_masked",
#          "/fs5/p_masi/krishar1/MIDL/B30f_B50f/soft_masked",
#          "/fs5/p_masi/krishar1/MIDL/C_D/hard_masked",
#          "/fs5/p_masi/krishar1/MIDL/C_D/soft_masked",
#          "/fs5/p_masi/krishar1/MIDL/STANDARD_LUNG/hard",
#          "/fs5/p_masi/krishar1/MIDL/STANDARD_LUNG/soft",
#          "/fs5/p_masi/krishar1/MIDL/STANDARD_BONE/hard",
#          "/fs5/p_masi/krishar1/MIDL/STANDARD_BONE/soft"]

paths = ["/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/B30f_B50f/hard_masked", 
"/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/B30f_B50f/soft_masked"]

c3d_path = "/home/local/VANDERBILT/krishar1/c3d-1.0.0-Linux-x86_64/bin/c3d"

# Look in every folder of the pid. Should be pid_multilabel.nii.gz. split using c3d command. 

for path in os.listdir(paths[1]):
    image_path = os.path.join(paths[1], path, "segmentations", path + "_multilabel.nii.gz")
    out_dir = os.path.join("/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/B30f_B50f", "soft_slices")
    out_file = os.path.join(out_dir, f'{path}_%03d.nii.gz')
    c3d_command = f'{c3d_path} {image_path} -slice z 0%:100% -oo {out_file}'
    print(c3d_command)
    os.system(c3d_command)

