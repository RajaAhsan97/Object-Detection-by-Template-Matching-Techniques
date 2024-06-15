"""
    Template matching

    -----------------------------------------------------------------
    |    Six template matching algorithms are implemented here:     |
    -----------------------------------------------------------------
    |    1. Sum of Square Difference             |  (SQDIFF)        |
    |    2. Normalized Sum of Square Difference  |  (SQDIFF_NORMED) |
    |    3. Cross correlation                    |  (CCORR)         |
    |    4. Normalized cross correlation         |  (CCORR_NORMED)  |
    |    5. Correlation coefficient              |  (COEFF)         |
    |    6. Normalized Correlation coefficient   |  (COEFF_NORMED)  |
    -----------------------------------------------------------------

    In SQDIFF and SQDIFF_NORMED minimum value indicate the better match
    between the template and image patch. Whereas in CCORR, CCORR_NORMED,
    COEFF and COEFF_NORMED the higher value indicates the better match.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

TM_dir = "Template Matching"
if not os.path.exists(TM_dir):
    os.mkdir(TM_dir)

image_nm = "1st.jpeg"
dir_nm = os.path.join(TM_dir, image_nm.split(".")[0])
# create directory for save the output of algorithm for image
if not os.path.exists(dir_nm):
    os.mkdir(dir_nm)


# function for finding the min and max value and their coordinated of the result image
def minMaxLoc(result):
    max_val, min_val = np.max(np.max(result)), np.min(np.min(result))
    coord_max, coord_min = np.where(result == max_val), np.where(result == min_val)
    max_loc, min_loc = (coord_max[1][0], coord_max[0][0]), (coord_min[1][0], coord_min[0][0])
    return min_val, max_val, min_loc, max_loc  

# Correlation coefficient
def COEFF(image, template, out, image_resol, temp_hei, temp_wid):
    for r in range(image_resol[0]):
        r_end = r + temp_hei
        if r_end > image_resol[0]:
            break
        for c in range(image_resol[1]):
            c_end = c + temp_wid
            if c_end > image_resol[1]:
                break
            ROI = np.array(image[r:r_end, c:c_end], dtype=float)

            temp_avg_diff = template - (np.sum(template)/(temp_wid*temp_hei))
            roi_avg_diff = ROI - (np.sum(ROI)/(temp_wid*temp_hei))

            out[r,c] = np.sum(np.multiply(roi_avg_diff, temp_avg_diff))
    return out

# Correlation coefficient Normalized
def COEFF_NORMED(image, template, out, image_resol, temp_hei, temp_wid):
    for r in range(image_resol[0]):
        r_end = r + temp_hei
        if r_end > image_resol[0]:
            break
        for c in range(image_resol[1]):
            c_end = c + temp_wid
            if c_end > image_resol[1]:
                break
            ROI = np.array(image[r:r_end, c:c_end], dtype=float)

            temp_avg_diff = template - (np.sum(template)/(temp_wid*temp_hei))
            roi_avg_diff = ROI - (np.sum(ROI)/(temp_wid*temp_hei))

            COEFF = np.sum(np.multiply(roi_avg_diff, temp_avg_diff))
            NORM = np.sqrt(np.sum(np.power(temp_avg_diff, 2)) * np.sum(np.power(roi_avg_diff, 2)))

            out[r,c] = np.divide(COEFF, NORM)
    return out

# Cross-correlation 
def CCORR(image, template, out, image_resol, temp_hei, temp_wid):

    # Sliding window of the patch region from top-left to top-right and top to bottom 
    for r in range(image_resol[0]):
        r_end = r + temp_hei
        if r_end > image_resol[0]:
            break
        
        for c in range(image_resol[1]):
            c_end = c + temp_wid
            if c_end > image_resol[1]:
                break
        
            ROI = np.array(image[r:r_end, c:c_end], dtype=float)
            CCORR = np.sum(np.multiply(ROI, template))
            # normalized cross correlation
            out[r,c] = CCORR
    return out

# Normalized cross-correlation
def CCORR_NORMED(image, template, out, image_resol, temp_hei, temp_wid):

    # Sliding window of the patch region from top-left to top-right and top to bottom 
    for r in range(image_resol[0]):
        r_end = r + temp_hei
        if r_end > image_resol[0]:
            break
        
        for c in range(image_resol[1]):
            c_end = c + temp_wid
            if c_end > image_resol[1]:
                break
        
            ROI = np.array(image[r:r_end, c:c_end], dtype=float)
            CCORR = np.sum(np.multiply(ROI, template))
            NORM = np.sqrt(np.multiply(np.sum(np.power(ROI, 2)), np.sum(np.power(template,2))))
            # normalized cross correlation
            out[r,c] = np.divide(CCORR, NORM)
    return out

# Sum of Square difference
def SQDIFF(image, template, out, image_resol, temp_hei, temp_wid):
    
    for r in range(image_resol[0]):
        r_end = r + temp_hei
        if r_end > image_resol[0]:
            break
        for c in range(image_resol[1]):
            c_end = c + temp_wid
            if c_end > image_resol[1]:
                break
            ROI = np.array(image[r:r_end, c:c_end], dtype=float)
            SQDIFF = np.sum(np.power(ROI-template , 2))
            out[r,c] = SQDIFF
    return out

# recheck normalized results (normalized values lies b/w 0-1)
def SQDIFF_NORMED(image, template, out, image_resol, temp_hei, temp_wid):
    for r in range(image_resol[0]):
        r_end = r + temp_hei
        if r_end > image_resol[0]:
            break
        for c in range(image_resol[1]):
            c_end = c + temp_wid
            if c_end > image_resol[1]:
                break
            ROI = np.array(image[r:r_end, c:c_end], dtype=float)
            SQDIFF = np.sum(np.power(ROI-template , 2))
            NORM = np.sqrt(np.multiply(np.sum(np.power(ROI, 2)) , np.sum(np.power(template, 2))))

            out[r,c] = np.divide(SQDIFF, NORM)
    return out

def matchTemplate(image, template, algorithm):
    image_resol = np.shape(image)
    temp_hei, temp_wid = template.shape
    
    out_img = np.zeros((image_resol[0]-temp_hei+1, image_resol[1]-temp_wid+1), dtype=float)

    if algorithm == "CCORR":
        time_start = time.time()
        rslt_img = CCORR(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")
        
        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = max_loc
        bottom_right = (top_left[0] + temp_wid, top_left[1] + temp_hei)
    elif algorithm == "CCORR_NORMED":
        time_start = time.time()
        rslt_img = CCORR_NORMED(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")
        
        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = max_loc
        bottom_right = (top_left[0] + temp_wid, top_left[1] + temp_hei)
    elif algorithm == "SQDIFF":
        time_start = time.time()
        rslt_img = SQDIFF(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")
        
        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = min_loc
        bottom_right = (top_left[0] + temp_wid, top_left[1] + temp_hei)
    elif algorithm == "SQDIFF_NORMED":
        time_start = time.time()
        rslt_img = SQDIFF_NORMED(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")
        
        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = min_loc
        bottom_right = (top_left[0] + temp_wid, top_left[1] + temp_hei)
    elif algorithm == "COEFF":
        time_start = time.time()
        rslt_img = COEFF(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")
        
        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = max_loc
        bottom_right = (top_left[0] + temp_wid, top_left[1] + temp_hei)
    elif algorithm == "COEFF_NORMED":
        time_start = time.time()
        rslt_img = COEFF_NORMED(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")
        
        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = max_loc
        bottom_right = (top_left[0] + temp_wid, top_left[1] + temp_hei)
        
    return rslt_img, top_left, bottom_right


# Template matching methods
temp_mtch_algos = {
                        1: "CCORR",
                        2: "CCORR_NORMED",
                        3: "SQDIFF",
                        4: "SQDIFF_NORMED",
                        5: "COEFF",
                        6: "COEFF_NORMED"
                    }
    
print("--------------------------------------")
print("Template matching algorithms")
print(" 1. Cross-Correlation (CCORR)\n 2. Normalized Cross-Correlation (CCORR_NORMED)\n 3. Square Difference (SQDIFF)\n 4. Normalized Square Difference (SQDIFF_NORMED)\n 5. Correlation coefficient (COEFF)\n 6. Normalized Correlation coefficient (COEFF_NORMED)")
algo = int(input("Select algorithm: "))
algo= temp_mtch_algos[algo]



# Read the source image
image = cv2.imread(image_nm)
rgb_image = image.copy()
image_resol = np.shape(image)

# convert source image to gray scale
channels = 1 if len(image_resol) == 2 else image_resol[-1]

if channels == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
elif channels == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

# extract the patch from source image
# 2nd.jpeg
#    image[15:33, 123:138], image[21:42, 211:233]
# 1st.jpeg

template = np.array(image[89:119, 136:166], dtype=float)

# save the template image
cv2.imwrite(os.path.join(dir_nm, "_ROI.jpg"), template)

# Template matching
result, top_left, bottom_right= matchTemplate(image, template, algo)

# indicate matched ROI on the image
cv2.rectangle(rgb_image, top_left, bottom_right, (0,255,0), 1)


# save the output to imaghe directory
save_out_img = os.path.join(dir_nm, algo + "_out.jpg")

cv2.imwrite(save_out_img, rgb_image)


fig, ax = plt.subplots(nrows=1, ncols=2)


ax[0].set_title(algo+"_output")
ax[0].imshow(result)

ax[1].set_title("Detected Region")
ax[1].imshow(rgb_image)

plt.savefig(os.path.join(dir_nm, algo + "_result.jpg"))
