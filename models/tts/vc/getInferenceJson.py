import os
import json

# Define the directories
folder = '/mntnfs/lee_data1/vcdata/VCTK'
gt_dir = folder + '/gt'
prompt_dir = folder + '/prompt'
source_dir =  folder + '/source'

# List files in each directory and sort them
gt_files = sorted(os.listdir(gt_dir))
prompt_files = sorted(os.listdir(prompt_dir))
source_files = sorted(os.listdir(source_dir))

# Initialize the list for test cases
test_cases = []

# Iterate over files in all directories simultaneously
for source_file, gt_file, prompt_file in zip(source_files, gt_files, prompt_files):
    # Extract UIDs and check if the last three digits match
    source_uid = source_file.split('.')[0]
    gt_uid = gt_file.split('.')[0]
    if source_uid[-3:] == gt_uid[-3:]:
        # Construct file paths
        source_wav_path = os.path.join(source_dir, source_file)
        target_wav_path = os.path.join(gt_dir, gt_file)
        prompt_wav_path = os.path.join(prompt_dir, prompt_file)

        # Create a dictionary for the test case
        test_case = {
            "uid": source_uid,  # or use gt_uid based on which is preferred
            "source_wav_path": source_wav_path,
            "target_wav_path": target_wav_path,
            "prompt_wav_path": prompt_wav_path
        }

        # Add the test case to the list
        test_cases.append(test_case)

# Save the test cases to a JSON file
zero_shot_json_file_path = '/mntnfs/lee_data1/vcdata/VCTK/zero_shot_json.json'
with open(zero_shot_json_file_path, 'w') as f:
    json.dump({"test_cases": test_cases}, f, indent=4)
