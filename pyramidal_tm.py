"""
    In this code template matching is performed by downscaling the image to
    one-fourth of its original resolution, which reduces the time consumption
    for template matching by the algorithm.

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

TM_dir = "Pyramidal_TM"
if not os.path.exists(TM_dir):
    os.mkdir(TM_dir)

image_nm = "1st.jpeg"
dir_nm = os.path.join(TM_dir, image_nm.split(".")[0])
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

def matchTemplate(image, template, algorithm, scale_fac):
    image_resol = np.shape(image)
    temp_hei, temp_wid = template.shape
    
    out_img = np.zeros((image_resol[0]-temp_hei+1, image_resol[1]-temp_wid+1), dtype=float)

    if algorithm == "CCORR":
        time_start = time.time()
        rslt_img = CCORR(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")
        
        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = tuple(np.multiply(max_loc, scale_fac))
        bottom_right = (top_left[0] + temp_wid*scale_fac, top_left[1] + temp_hei*scale_fac)
    elif algorithm == "CCORR_NORMED":
        time_start = time.time()
        rslt_img = CCORR_NORMED(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")
                
        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = tuple(np.multiply(max_loc, scale_fac))
        bottom_right = (top_left[0] + temp_wid*scale_fac, top_left[1] + temp_hei*scale_fac)
    elif algorithm == "SQDIFF":
        time_start = time.time()
        rslt_img = SQDIFF(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")

        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = tuple(np.multiply(min_loc, scale_fac))
        bottom_right = (top_left[0] + temp_wid*scale_fac, top_left[1] + temp_hei*scale_fac)
    elif algorithm == "SQDIFF_NORMED":
        time_start = time.time()
        rslt_img = SQDIFF_NORMED(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")

        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = tuple(np.multiply(min_loc, scale_fac))
        bottom_right = (top_left[0] + temp_wid*scale_fac, top_left[1] + temp_hei*scale_fac)
    elif algorithm == "COEFF":
        time_start = time.time()
        rslt_img = COEFF(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")
        
        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = tuple(np.multiply(max_loc, scale_fac))
        bottom_right = (top_left[0] + temp_wid*scale_fac, top_left[1] + temp_hei*scale_fac)
    elif algorithm == "COEFF_NORMED":
        time_start = time.time()
        rslt_img = COEFF_NORMED(image, template, out_img, image_resol, temp_hei, temp_wid)
        time_end = time.time()
        print(algorithm,  ": time consumption: ", np.round(time_end - time_start, 2), "sec")
        
        min_val, max_val, min_loc, max_loc = minMaxLoc(rslt_img)
        
        top_left = tuple(np.multiply(max_loc, scale_fac))
        bottom_right = (top_left[0] + temp_wid*scale_fac, top_left[1] + temp_hei*scale_fac)
        
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

scale_factor = 4

# Read source image
image = cv2.imread(image_nm)
rgb_image = image.copy()
image_resol = np.shape(image)

# convert source image to gray scale
channels = 1 if len(image_resol) == 2 else image_resol[-1]
if channels == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
elif channels == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)


# crop template from the image
crop_r1, crop_r2 = 89, 119
crop_c1, crop_c2 = 136, 166
template = np.array(image[crop_r1:crop_r2, crop_c1:crop_c2], dtype=float)
temp_hei, temp_wid = template.shape

# downscale image resolution to one-fourth
scale_wid, scale_hei = int(np.round(image_resol[1]/scale_factor)), int(np.round(image_resol[0]/scale_factor))
resize_img = cv2.resize(image, (scale_wid, scale_hei))

# get template from the downsampled image
down_samp_r1, down_samp_r2 = int(np.round(crop_r1/scale_factor)), int(np.round(crop_r2/scale_factor))
down_samp_c1, down_samp_c2 = int(np.round(crop_c1/scale_factor)), int(np.round(crop_c2/scale_factor))
resize_temp = np.array(resize_img[down_samp_r1:down_samp_r2, down_samp_c1:down_samp_c2], dtype=float)
temp_scale_hei, temp_scale_wid = resize_temp.shape

# template matching on downscaled image
result, top_left, bottom_right = matchTemplate(resize_img, resize_temp, algo, scale_factor)

# Draw matched region on th image
cv2.rectangle(rgb_image, top_left, bottom_right, (0,255,0), 1)

fig, ax = plt.subplots(nrows=4, ncols=2)

ax[0,0].imshow(image)
ax[0,0].set_title("Original image")
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

ax[0,1].imshow(resize_img)
ax[0,1].set_title("Downscaled image (1/4th)")
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

ax[1,0].imshow(template)
ax[1,0].set_title("template")
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

ax[1,1].imshow(resize_temp)
ax[1,1].set_title("Downscaled template(1/4th)")
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

ax[2,0].imshow(image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
ax[2,0].set_title("Template Found")
ax[2,0].set_xticks([])
ax[2,0].set_yticks([])

ax[2,1].imshow(result)
ax[2,1].set_title("downsampled result")
ax[2,1].set_xticks([])
ax[2,1].set_yticks([])

ax[3,0].imshow(rgb_image)
ax[3,0].set_title("TM Original image")
ax[3,0].set_xticks([])
ax[3,0].set_yticks([])

plt.savefig(os.path.join(dir_nm, algo + "_result.jpg"))

cv2.imwrite(os.path.join(dir_nm, algo + "_out.jpg"), rgb_image)
