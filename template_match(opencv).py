import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

temp_mtch_methods = {
                        1: 'cv2.TM_CCOEFF',
                        2: 'cv2.TM_CCOEFF_NORMED',
                        3: 'cv2.TM_CCORR',
                        4: 'cv2.TM_CCORR_NORMED',
                        5: 'cv2.TM_SQDIFF',
                        6: 'cv2.TM_SQDIFF_NORMED'
                    }

print("----------- Template Matching Methods -----------")
print(" 1. cv2.TM_CCOEFF \n 2. cv2.TM_CCOEFF_NORMED \n 3. cv2.TM_CCORR \n 4. cv2.TM_CCORR_NORMED \n 5. cv2.TM_SQDIFF \n 6. cv2.TM_SQDIFF_NORMED")
method = int(input("Select method for template matching: "))

image = cv2.imread("1st.jpeg")
rgb_img = image.copy()
image_resol = np.shape(image)

channels = 1 if len(image_resol) == 2 else image_resol[-1] 

if channels == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
elif channels == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

# crop the region of image that can be used as template
template = image[89:119, 136:166]
template_resol = np.shape(template)
temp_hei, temp_wid = template_resol[0], template_resol[1]
cv2.imwrite("crop_im.jpg", template)

# template matching
result = cv2.matchTemplate(image, template, eval(temp_mtch_methods[method]))
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)


# determine coordinated of matched region
top_left = max_loc
bottom_right = (top_left[0] + temp_wid, top_left[1] + temp_hei)

# Identify matched regio on image
cv2.rectangle(rgb_img, top_left, bottom_right, (0, 255,0), 1)

cv2.imwrite("matched_region.jpg", rgb_img)

plt.imshow(result)
plt.show()
