import os
import shutil
# Walk through ./data/ and find folders containing whole-body, attenuation corrected PET dicoms.

data_dir = './data/acrin-nsclc-fdg-pet/manifest-1580838164030/ACRIN-NSCLC-FDG-PET/'

ndirs = 0
good_pet_paths = []
good_ct_paths = []

n_candidate_pet = []

for root, dirs, files in os.walk(data_dir):
    for dir in dirs:
        if 'PETCT' in dir or 'PT CT' in dir:
            ndirs += 1
            # For each study we need to identify ONE PET folder and ONE CT folder.
            # We're gonna do this based on string matching and counting dicoms
            # We want the CT dicom count to match exactly the PET dicom count
            # We want the PET folder to not be the uncorrected (No AC) PET dicoms
            for study_root, study_dirs, study_files in os.walk(os.path.join(root, dir)):
                dicom_counts = [len(os.listdir(os.path.join(study_root, dir))) for dir in study_dirs]

                candidate_pet_paths = []
                candidate_ct_paths = []

                for study_dir in study_dirs:
                    study_dir_lower = study_dir.lower()
                    if 'pet' in study_dir_lower and \
                        not 'nac' in study_dir_lower and \
                        not 'uncorrected' in study_dir_lower and \
                        not 'no ac' in study_dir_lower:
                        candidate_pet_paths.append(os.path.join(study_root, study_dir, ''))
                    if 'ct ' in study_dir_lower:
                        candidate_ct_paths.append(os.path.join(study_root, study_dir, ''))

                n_candidate_pet.append(len(candidate_pet_paths))
                # If we've found exactly 1 candidate dir for PET and CT, and they each
                # contain the same number of dicoms, add 'em to the list of good directories
                if len(candidate_pet_paths) == 1 and \
                    len(candidate_ct_paths) == 1 and \
                    len(os.listdir(candidate_pet_paths[0])) == len(os.listdir(candidate_ct_paths[0])):
                    good_ct_paths.append(candidate_ct_paths[0])
                    good_pet_paths.append(candidate_pet_paths[0])
                        

# Using this approach, we get 97 matched PET and CT folders out of a possible 275 directories.
# Thats good enough for now. If we end up needing more training data, I can come back and try to
# increase our yield here.
print(str(ndirs))
print(str(len(good_pet_paths)))
print(str(len(good_ct_paths)))

# Now, copy our cleaned set of good CT and PET directories to a cleaned location with a clean file structure
out_dir = './data/acrin-nsclc-fdg-cleaned/'
if not os.path.exists(out_dir): os.makedirs(out_dir) 

for ct_path, pet_path in zip(good_ct_paths, good_pet_paths):
    path_parts = ct_path.split('/')

    subject_id = path_parts[5] + '_' + path_parts[6]

    this_subject_ct_path = out_dir + '/' + subject_id + '/CT/'
    this_subject_pet_path = out_dir + '/' + subject_id + '/PET/'
    
    if not os.path.exists(this_subject_ct_path): os.makedirs(this_subject_ct_path)
    if not os.path.exists(this_subject_pet_path): os.makedirs(this_subject_pet_path)

    for f in os.listdir(ct_path):
        shutil.copy2(os.path.join(ct_path, f), os.path.join(this_subject_ct_path, f))

    for f in os.listdir(pet_path):
        shutil.copy2(os.path.join(pet_path, f), os.path.join(this_subject_pet_path, f))