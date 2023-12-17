import os
import shutil

# Walk through ./data/ and find folders containing whole-body, attenuation corrected PET dicoms.
# TODO: clean this up

data_dir = '/home/daniel/datasets/manifest-1580838164030/ACRIN-NSCLC-FDG-PET/'
out_dir = '/home/daniel/datasets/ACRIN-NSCLC-FDG-PET-cleaned/'

ndirs = 0
good_pet_paths = []
good_ct_paths = []

n_candidate_pet = []

for patient in os.listdir(data_dir):
    patient_dir = data_dir + patient + '/'

    for timepoint in os.listdir(patient_dir):
        timepoint_dir = patient_dir + timepoint + '/'

        candidate_pet_paths = []
        candidate_ct_paths = []

        for series in os.listdir(timepoint_dir):
            series_dir = timepoint_dir + series + '/'

            series_lower = series.lower()
            if 'pet' in series_lower and \
                    'nac' not in series_lower and \
                    'uncorrected' not in series_lower and \
                    'no ac' not in series_lower:
                candidate_pet_paths.append(series_dir)
            elif 'ct ' in series_lower:
                candidate_ct_paths.append(series_dir)
            else:
                print("We can delete this series: " + series_dir)
                for f in os.listdir(series_dir):
                    os.remove(os.path.join(series_dir, f))

        if len(candidate_pet_paths) == 1 and \
                len(candidate_ct_paths) == 1 and \
                len(os.listdir(candidate_pet_paths[0])) == len(os.listdir(candidate_ct_paths[0])):
            good_ct_paths.append(candidate_ct_paths[0])
            good_pet_paths.append(candidate_pet_paths[0])

print("Found " + str(len(good_pet_paths)) + " useful PET directories.")
print("Found " + str(len(good_ct_paths)) + " useful CT directories.")

# Now, copy our cleaned set of good CT and PET directories to a cleaned location with a clean file structure
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for ct_path, pet_path in zip(good_ct_paths, good_pet_paths):
    path_parts = ct_path.split('/')

    patient_id = path_parts[6]
    timepoint_id = path_parts[7].replace(' ', '_')

    subject_id = patient_id + '_' + timepoint_id
    print(subject_id)

    this_subject_ct_path = out_dir + '/' + subject_id + '/CT/'
    this_subject_pet_path = out_dir + '/' + subject_id + '/PET/'

    if not os.path.exists(this_subject_ct_path): os.makedirs(this_subject_ct_path)
    if not os.path.exists(this_subject_pet_path): os.makedirs(this_subject_pet_path)

    for f in os.listdir(ct_path):
        shutil.copy2(os.path.join(ct_path, f), os.path.join(this_subject_ct_path, f))

    for f in os.listdir(pet_path):
        shutil.copy2(os.path.join(pet_path, f), os.path.join(this_subject_pet_path, f))
