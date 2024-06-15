This repository contains object detection by mean of template matching techniques. Template matching includes the source image and a patch of the image, the patch id the subregion of the images
which will be found in the image. The patch is slided over the image from left to right and top to bottom by one pixel, and the result is computed for each region of image on which the patch is overlayed 
by performing the operation just like convolution. For template matching, various techniques are proposed as mentioned below,  which will be performed between the patch and the image region. t

1.   Sum of Square Difference (SQDIFF)
2.   Normalized Sum of Square Difference (SQDIFF_NORMED)
3.   Cross correlation (CCORR)
4.   Normalized cross correlation (CCORR_NORMED)
5.   Correlation coefficient (COEFF)
6.   Normalized correlation coefficient (COEFF_NORMED)  

For better match SQDIFF and SQDIFF_NORMED will result in minimum score, whereas for other techniques maximum score represents the better match.

Furthermore the Pyramidal template matching are also implemented which includes all of the aboved mentioned techniques, but in this template matching is performed on the sampled down image.
The source image is scaled down to one-fourth of its original resolution, which leads to less computational time by the algorithms.

For reference the above techniques are also implemented by using opencv. (https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html)

Image:
![1st](https://github.com/RajaAhsan97/Object-Detection-by-Template-Matching-Techniques/assets/155144523/cd020bd5-70c6-4987-8ebc-3cba4c7cca85)

ROI: (to be found in the image)
![_ROI](https://github.com/RajaAhsan97/Object-Detection-by-Template-Matching-Techniques/assets/155144523/1f56a4c1-d0d6-447c-a59f-afe7fe0d514f)

Template matching:
![SQDIFF_NORMED_result](https://github.com/RajaAhsan97/Object-Detection-by-Template-Matching-Techniques/assets/155144523/bec9dd63-d8ce-497f-8e84-a1cc591271c2)

Pyramidal Template matching:
![SQDIFF_NORMED_result](https://github.com/RajaAhsan97/Object-Detection-by-Template-Matching-Techniques/assets/155144523/68134619-61b3-491e-a37f-2be39c6a6bd3)

