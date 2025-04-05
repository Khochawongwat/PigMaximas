**Folder Structure Requirement**
To use this system, ensure that your file structure follows this format:

This is inside the 'cnn' folder.
batches/
    ├── batch_A/
    │   ├── pig_A/
    │   │   ├── frame_2_depth_crop.png
    │   │   ├── frame_2_rgb_crop.png
    │   ├── pig_B/
    │       ├── frame_2_depth_crop.png
    │       ├── frame_2_rgb_crop.png
    ├── batch_B/
        ├── pig_A/
        │   ├── frame_2_depth_crop.png
        │   ├── frame_2_rgb_crop.png
        ├── pig_B/
            ├── frame_2_depth_crop.png
            ├── frame_2_rgb_crop.png
main.ipynb

**Folder Structure**:

    batches/ is the root directory containing all batch folders.

    Each batch (e.g., batch_A) contains folders for individual pigs (e.g., pig_A, pig_B).

    Inside each pig folder, include only image files (e.g., frame_2_depth_crop.png, frame_2_rgb_crop.png). Do not rename your image files. Maintain the original file names.

**Important Notes**:

Keep image names as they are (e.g., frame_2_depth_crop.png, frame_2_rgb_crop.png).

Do not rename images to ensure they are correctly identified and processed.

After uploading additional pigs and batches. You must go to the cnn\main.ipynb and change the size of train_keys and val_keys.

Lastly for the pig's weight. Go to this line: 

for (key, df), w in zip(total.items(), [15.21828, 17.3282571, 17.0191098, 16.7612445]):
    total[key]["weight"] = w

Add more weights according to the order that you put your pig in.
So if you have 10 pigs. You must have 10 weights. Just like so.
for (key, df), w in zip(total.items(), [15.21828, 17.3282571, 17.0191098, 16.7612445, 15.21828, 17.3282571, 17.0191098, 16.7612445, 15.21828, 17.3282571]):
    total[key]["weight"] = w