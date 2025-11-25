#load libraries
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-789a4674-4ff1-2bbc-4f59-c4540006f305"
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import multiprocessing as mp
from traceback import format_exception

def now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

ignored_classes = {'Label'}

# Function to convert filename timestamp to datetime object
def convert_timestamp(filename):
    # ignore extension (last 4 chars)
    return datetime.strptime(filename[:-4], "%Y%m%d_%H%M%S")


# Color coding for object classes
class_colors = {
    "Dust": (255, 0, 0),  # Red
    "Amorphous": (0, 255, 0),  # Green
    "Crystal": (255, 255, 0)  # Cyan
}

def calculate_crystal_pixels(image, threshold_to_use=None):
    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect circles on the raw grayscale image
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,         # inverse ratio of accumulator resolution
        minDist=80,     # min dist between circle centers
        param1=100,     # higher threshold for Canny edge detector
        param2=20,      # accumulator threshold (lower is more circles)
        minRadius=50, 
        maxRadius=128
    )

    #------ set boundary for cicular well -----#
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]

        r = max(0, r - 40) #to remove well exterior

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        roi = cv2.bitwise_and(gray, gray, mask=mask)

        #--------- normalise pixel intensity for each image ---------#
        mask = mask.astype(bool)
        roi_max = roi[mask].max()
        roi_min = roi[mask].min() #ignores background and will normalise to pixels >0 e.g. the well interior

        roi_float = roi.astype(np.float32)

        w = (roi_float-roi_min)/(roi_max-roi_min)
        w[w < 0] = 0.0
        
        #scale back to 255 and convert for saving image
        roi_normalised = (w * 255).astype(np.uint8)
        
        # ------------ Gaussian Filter prior to Otsu thresh ------------#
        roi_blurred = cv2.GaussianBlur(roi_normalised, ksize=(21, 21), sigmaX=0)
        roi_for_otsu = roi_blurred.copy()
        # --- Otsu's Thresholding ---
        if threshold_to_use is None:
            otsu_threshold, binary = cv2.threshold(
                roi_for_otsu, 
                0, 
                255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                roi_for_otsu, 
                threshold_to_use, #use set thresh 
                255, 
                cv2.THRESH_BINARY
            )

        #------ normalise pixel count by area -----#
        crystal_pixels_raw = cv2.countNonZero(binary) #total crystal pixel count
        total_well_pixels = np.count_nonzero(mask) #total pixels in mask

        crystal_pixel_ratio = crystal_pixels_raw/total_well_pixels

        return crystal_pixel_ratio, (otsu_threshold if threshold_to_use is None else threshold_to_use)

    else:
        # fallback if no circle is detected
        return 0.0


def assess_experiment(image_folder: str):
    try:
        _assess_experiment(image_folder)
    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(f'error running {image_folder}\n')
            for line in format_exception(type(e), e, e.__traceback__):
                f.write(f'{line}')

def _assess_experiment(image_folder:str):
    wells_detection_model_path= '/home/lero/idrive/cmac/DDMAP/Image_analysis/Code/wells_model/Results/fold_3/weights/best.pt'
    crystal_detection_model_path = '/home/lero/idrive/cmac/DDMAP/Image_analysis/Code/cryst_amorphous_model/Results/fold_1/weights/best.pt'
    wells_detection_model = YOLO(wells_detection_model_path)
    crystal_detection_model = YOLO(crystal_detection_model_path)
    pid = os.getpid()
    output_folder = os.path.join(image_folder, "new_results")
    os.makedirs(output_folder, exist_ok=True)

    #get sorted images
    images = sorted(img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg')))
    if not images:
        print(f'no images found in {image_folder}, skipping')
        return

    #get the first image timestamp
    assert len(images)
    first_image_timestamp = convert_timestamp(images[0])
    first_image_timestamp_str = first_image_timestamp.strftime("%d-%m-%Y %H:%M")

    wells = list(range(96))
    well_data = {wi: [] for wi in wells}
    stability_results = {wi:{'Timestamp':None, 'Class': None} for wi in wells}
    WINDOW = 80
    fixed_threshold = {} #for otsu binarisation
    crystallised_state = {wi: False for wi in wells}

    #---------read and process images-------------#
    for idx, image_name in enumerate(tqdm(images, unit='images')):
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path) #image of well plate

        results = wells_detection_model(img, verbose=False)
        located_wells = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                cls_name = result.names[cls_id]
                conf = round(float(box.conf), 2)

                if cls_name in ignored_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                located_wells.append((x1, y1, x2, y2, center_x, center_y, cls_name, conf)) 
        
        normalised_wells = []
        
        #normalise grid based on well [0] and map all wells into grid positions
        if located_wells:
            all_confs = sorted([o[-1] for o in located_wells], reverse=True)

            x = [o[4] for o in located_wells if o[-1] > 0.75]
            y = [o[5] for o in located_wells if o[-1] > 0.75]
            left = min(x)
            right = max(x)
            top = min(y)
            bottom = max(y)
            width = right - left
            height = bottom - top
            reference_well = min(located_wells, key=lambda o: (o[5], o[4])) #since origin of image is (0,0)
            ref_x, ref_y = reference_well[4], reference_well[5]

            for obj in located_wells:
                x1, y1, x2, y2, center_x, center_y, cls_name, conf = obj
                midp_x = (x1+x2)/2
                midp_y = (y1+y2)/2
                x_coord = int((midp_x - left)*12/width/1.03)
                y_coord = int((midp_y - top)*8/height)
                x_coord = min(11, max(0, x_coord)) #set min and max rows from 0,11
                y_coord = min(7, max(0, y_coord)) #set min and max columns from 0,7
                normalised_wells.append((x1, y1, x2, y2, center_x, center_y, cls_name, conf, x_coord, y_coord))
            
            #keep only the best detections per grid 
            best_detections = {}
            for obj in normalised_wells:
                x1, y1, x2, y2, center_x, center_y, cls_name, conf, x_coord, y_coord = obj
                well_key = (x_coord, y_coord)
                if well_key not in best_detections or conf > best_detections[well_key][7]:
                    best_detections[well_key] = obj
            normalised_wells = list(best_detections.values())

        #-----take patches------#
        for patch in normalised_wells:
            x1, y1, x2, y2, center_x, center_y, cls_name, conf, x_coord, y_coord = patch
            well_key = x_coord*8+y_coord #0-95 indexing
                    
            #crop the patch using the coordinates from grid position
            well_crop = img[y1:y2, x1:x2]
            
            ##--possible to save well image here--##

            crystal_results = crystal_detection_model(well_crop, verbose=False)
            
            #extract top classification prediction
            top_cls_name=None
            top_conf= 0
            for res in crystal_results:
                for crop in res.boxes:
                    patch_cls_id = int(crop.cls)
                    patch_conf = float(crop.conf)
                    if patch_conf > top_conf:
                        top_conf = patch_conf
                        top_cls_name = res.names[patch_cls_id]

            fixed_thresh = fixed_threshold.get(well_key)
            if top_cls_name == 'Crystal':
                if fixed_thresh is None:
                    crystal_number_pixels, new_thresh = calculate_crystal_pixels(well_crop, threshold_to_use=None)
                    fixed_threshold[well_key] = new_thresh
                else:
                    crystal_number_pixels, _ = calculate_crystal_pixels(well_crop, threshold_to_use=fixed_thresh)
            else: # Dust or Amorphous
                # Retain last known pixel count for non-crystals
                if well_data[well_key]:
                    crystal_number_pixels = well_data[well_key][-1]['crystal_number_pixels']
                else:
                    crystal_number_pixels = 0
                        
            #store result
            well_data[well_key].append({
                "image": image_name,
                "crop_coords": (x1, y1, x2, y2),
                "class": top_cls_name,
                "conf": top_conf,
                "crystal_number_pixels": crystal_number_pixels
            })   

        #----annotate every 50th image for a visual check----#
        if idx % 50 == 0 and normalised_wells:
            annotated_img = img.copy()
            for patch_info in normalised_wells:
                x1, y1, x2, y2, center_x, center_y, cls_name, conf, x_coord, y_coord = patch_info
                well_key = x_coord*8+y_coord  # 0-95 index
        
                if well_data.get(well_key):
                    top_result = well_data[well_key][-1]  # last classification for this image
                    label = f"Well {well_key}: {top_result['class']}, ({top_result['conf']:.2f})"
        
                    # Draw rectangle and label
                    color = class_colors.get(top_result['class'], (255, 255, 255))  # default white
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
            annotated_path = os.path.join(output_folder, f"annotated_{idx:04d}_{image_name}")
            cv2.imwrite(annotated_path, annotated_img)
            #print(f"✅ Saved annotated frame #{idx}: {annotated_path}")


        #----------- calculate well stability over time ----------#
        for well_key in wells:
            entries = well_data[well_key]
            if stability_results[well_key]['Class'] is None and len(entries) >= WINDOW:
                for i in range(len(entries) - WINDOW + 1):
                    current_class = entries[i]["class"]
                    if current_class == 'Crystal':
                        next_classes = [e['class'] for e in entries[i:i+WINDOW]]

                        if all(c=='Crystal' for c in next_classes):
                            stability_results[well_key]["Timestamp"] = entries[i]['image']
                            stability_results[well_key]['Class'] = 'Unstable'
                            break

    
    for well_key, entries in well_data.items():
        rows = []
        for entry in entries:
            rows.append({
                "well": well_key,
                "timestamp": entry["image"],
                "class": entry["class"],
                "crystal_number_pixels": entry["crystal_number_pixels"]
            })
        #if a well is not detected
        if not rows:
            continue #skip to next well key
    
        well_df = pd.DataFrame(rows)
        well_df["Timestamp_dt"] = well_df["timestamp"].apply(lambda fn: convert_timestamp(fn))
        
        well_df['1 week rolling average'] = well_df["crystal_number_pixels"].rolling(84).mean()
        
        #--create filtered columns for plotting crystal growth----#
        unstable_image = stability_results[well_key]["Timestamp"]
        well_df['plotted_crystal_number_pixels'] = 0
        well_df['plotted_rolling_average'] = float('nan')
        if unstable_image is not None:
            #find start time of 80 frame stability wiondow
            unstable_time_dt = convert_timestamp(unstable_image)
            mask = well_df["Timestamp_dt"] >= unstable_time_dt
            #apply raw data where confirmed stable
            well_df.loc[mask, 'plotted_crystal_number_pixels'] = well_df.loc[mask, "crystal_number_pixels"]
            well_df.loc[mask, 'plotted_rolling_average'] = well_df.loc[mask, '1 week rolling average']

        well_df_path = os.path.join(output_folder, f"well_{well_key}_data.csv")
        well_df.to_csv(well_df_path, index=False)
        #print("well data saves as csv")
        
        #-----plot well stability and mean pixel intensity -------#
        plt.figure(figsize=(4,3), dpi=200)
        
        #prep
        #well_df["Timestamp_dt"] = well_df["timestamp"].apply(lambda fn: convert_timestamp(fn))
        rel_time = well_df["Timestamp_dt"].astype('int64') / (10**9) / 3600 / 24
        cryst_or_no = well_df['class'].map(lambda c: 1 if c == "Crystal" else 0)
        if not len(rel_time):
            continue
        rel_time -= rel_time[0]

        #plot cryst vs time
        plt.plot(rel_time, cryst_or_no, 'C0')
        plt.xlabel('Time (days)')
        plt.yticks([0,1], ['No', 'Yes'], color='C0')
        plt.xticks(np.arange(0,91, step=10))
        plt.xlim(0,90)
        plt.ylabel('crystallised?', color='C0')

        #plot crystal growth
        ax2 = plt.twinx()
    
        ax2.plot(rel_time, well_df["plotted_crystal_number_pixels"], 'C1', label='crystal/background ratio')
        ax2.plot(rel_time, well_df['plotted_rolling_average'], 'C2', label='1 week rolling average')
        ax2.set_ylabel('Normalised crystal/ background ratio', color = 'C1')
        ax2.set_ylim(bottom=0, top=1)

        #draw cryst marker if exists
        unstable_image = stability_results[well_key]["Timestamp"]
        if unstable_image is not None:
            unstable_time = convert_timestamp(unstable_image)
            unstable_hours = (unstable_time - well_df["Timestamp_dt"].iloc[0]).total_seconds() / 3600 / 24
            plt.title(f'Crystallises after {unstable_hours:.2f} days')
            plt.axvline(unstable_hours, color='k', linestyle='--', lw=2)
        else:
            plt.title('Infinitely stable')

        ax2.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'well_{well_key}_stability_figure.png'))
        plt.close()
        

    stability_rows = []
    for well_key, info in stability_results.items():
        stability_rows.append({
            "well": well_key,
            "unstable_timestamp": info["Timestamp"],
            "status": info["Class"] if info["Class"] else "Stable"
        })
    
    stability_df = pd.DataFrame(stability_rows)
    
    stability_df_path = os.path.join(output_folder, "stability_results.csv")
    stability_df.to_csv(stability_df_path, index=False)
    print("✅ stability_results saved as CSV")

print('hopefully done')                                  

if __name__ == '__main__':

    file_path = '/home/lero/idrive/cmac/DDMAP/Stability studies'
    # folders = ['Image_analysis_test']
    folders = ['40_C_75_RH',
               '40_C_0_RH',
               '30_C_30_RH'
              ]
    all_imagefolders = []
    for folder in folders:
        folder_path = os.path.join(file_path, folder)
        #Image pathway
        for api in os.listdir(folder_path):
            image_folder = os.path.join(folder_path, api)
            if not os.path.isdir(image_folder):
                continue
            all_imagefolders.append(image_folder)


    nproc = 64
    with mp.Pool(nproc) as pool:
        n = len(all_imagefolders)
        chunks = max(1, n/nproc)
        done = 0
        for _ in pool.imap(assess_experiment, all_imagefolders):
            done += 1
            perc = done * 100.0 / n
            print(f'[main :: {now()}] {int(perc)}% complete.')

