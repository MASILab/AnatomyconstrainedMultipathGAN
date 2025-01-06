import os 
import shutil 
from tqdm import tqdm 

#Run TotalSeg all tissues. Run TotalSeg tissue types as well. 

#Write trial code to move all the files into corresponding folders. 
def make_data():
    siemens_hard = "/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/B30f_B50f/soft_masked"
    niftis = os.listdir(siemens_hard)

    for file in tqdm(niftis):
        file_path = os.path.join(siemens_hard, file)
        
        #Check if the file is a file or not.

        if os.path.isfile(file_path):
            folder_name = file.split(".nii.gz")[0]
            folder_path = os.path.join(siemens_hard, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            shutil.move(file_path, folder_path)

def run_totalseg(input_dir):
    #Run TotalSeg and tissue types to obtain labels"

    for directory in tqdm(os.listdir(input_dir)):
        main_file = os.listdir(os.path.join(input_dir, directory))
        for file in main_file:
            if file.endswith(".nii.gz"):
                nift_file = file
                nifti = os.path.join(input_dir, directory, nift_file)
                output_dir = os.path.join(input_dir, directory, "segmentations")
                os.makedirs(output_dir, exist_ok=True)
                print(f"TotalSegmentator -i {nifti} -o {output_dir}") #All structures 
                os.system(f"TotalSegmentator -i {nifti} -o {output_dir}")
                print(f"TotalSegmentator -i {nifti} -o {output_dir} -ta tissue_types") #Tissue types model
                os.system(f"TotalSegmentator -i {nifti} -o {output_dir} -ta tissue_types")




run_totalseg("/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/B30f_B50f/hard_masked")
# run_totalseg("/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/B30f_B50f/soft_masked")
# run_totalseg("/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/C_D/hard_masked")
# run_totalseg("/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/C_D/soft_masked")
# run_totalseg("/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/STANDARD_LUNG/hard")
# run_totalseg("/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/STANDARD_LUNG/soft")
# run_totalseg("/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/STANDARD_BONE/hard")
# run_totalseg("/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/STANDARD_BONE/soft")


