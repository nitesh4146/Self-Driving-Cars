from __future__ import print_function
from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

global combined_sobel, img_hls_white_yellow_bin, b_ch_out, sxbinary, prev_pts_left, prev_pts_right, slide_count

prev_pts_left = 0
prev_pts_right = 0
slide_count = 0

def calibrate():
    obj = np.zeros((9*6,3), np.float32)
    obj[:,0:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    objpoints = []
    imgpoints = []
    filelist = glob.glob('./camera_cal/*.jpg')

    C = cv2.imread(filelist[0])
    img_size = C.shape[1::-1]

    for file in filelist:
        I = cv2.imread(file)
        gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        if ret == True:
            objpoints.append(obj)
            imgpoints.append(corners)
#             cv2.drawChessboardCorners(I, (9,6), corners, ret)
#             cv2.imshow('img', I)
#             cv2.waitKey(500)
#     cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return mtx, dist

mtx, dist = calibrate()
print(">> Camera Calibrated")

def get_warped(img):
    global mtx, dist, combined_sobel, img_hls_white_yellow_bin, b_ch_out, sxbinary
    img = cv2.undistort(img, mtx, dist, None, mtx)
    
#     Sobel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    l_ch = lab[:,:,0]
    
    sobelx = cv2.Sobel(l_ch,cv2.CV_64F,1,0,ksize=15)
    sobely = cv2.Sobel(l_ch,cv2.CV_64F,0,1,ksize=15)
    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)
    
    sobelx_scaled = np.uint8(255*sobelx/np.max(sobelx))
    sobely_scaled = np.uint8(255*sobely/np.max(sobely))
    
    sobel_x_binary = np.zeros_like(sobelx)
    thresh_sobelx = [30, 120]
    sobel_x_binary[(sobelx_scaled > thresh_sobelx[0]) & (sobelx_scaled <= thresh_sobelx[1])] = 1
    
    sobel_y_binary = np.zeros_like(sobely)
    thresh_sobely = [20, 120]
    sobel_y_binary[(sobely_scaled > thresh_sobely[0]) & (sobely_scaled <= thresh_sobely[1])] = 1
    
    sobelxy = np.sqrt(sobelx**2 + sobely**2)
    mag_xy = np.uint8(255*sobelxy/np.max(sobelxy))
    sobel_mag_xy = np.zeros_like(sobelxy)
    thresh_mag = [80, 200]
    sobel_mag_xy[(mag_xy > thresh_mag[0]) & (mag_xy <= thresh_mag[1])] = 1
    
    dir_sxy = np.arctan2(sobelx_abs, sobely_abs)
    sobel_dir_xy = np.zeros_like(sobelxy)
    thresh_dir = [np.pi/4, np.pi/2]
    sobel_dir_xy[(dir_sxy > thresh_dir[0]) & (dir_sxy <= thresh_dir[1])] = 1
    
    combined_sobel = np.zeros_like(sobel_dir_xy)
    combined_sobel[(sobel_x_binary == 1) | ((sobel_y_binary == 1) & (sobel_mag_xy == 1) & (sobel_dir_xy == 1))] = 1
    
#     HLS
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_hls_yellow_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_yellow_bin[((hls_img[:,:,0] >= 15) & (hls_img[:,:,0] <= 35))
                 & ((hls_img[:,:,1] >= 30) & (hls_img[:,:,1] <= 204))
                 & ((hls_img[:,:,2] >= 115) & (hls_img[:,:,2] <= 255))                
                ] = 1
    
    # Compute a binary thresholded image where white is isolated from HLS components
    img_hls_white_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_white_bin[((hls_img[:,:,0] >= 0) & (hls_img[:,:,0] <= 255))
                 & ((hls_img[:,:,1] >= 200) & (hls_img[:,:,1] <= 255))
                 & ((hls_img[:,:,2] >= 0) & (hls_img[:,:,2] <= 255))                
                ] = 1
    
    # Now combine both
    img_hls_white_yellow_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1  
    
    #     B channel from LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_ch = lab[:,:,2]
    thres = [80, 113]
    b_ch_out= np.zeros_like(b_ch)
    b_ch_out[(b_ch > thres[0]) & (b_ch <= thres[1])] = 1

    color_binary = np.dstack(( np.zeros_like(combined_sobel), img_hls_white_yellow_bin, combined_sobel)) * 255
    sxbinary = np.zeros_like(combined_sobel)
    sxbinary[(combined_sobel ==1 ) | (img_hls_white_yellow_bin == 1) | (b_ch_out == 1)] = 1

    binary_t = sxbinary #get_threshold(I)
    (bottom_px, right_px) = (sxbinary.shape[0] - 1, sxbinary.shape[1] - 1) 
    src = np.array([[210,bottom_px],[595,450],[690,450], [1110, bottom_px]], np.float32)
    dst = np.array([[200, bottom_px], [200, 0], [1000, 0], [1000, bottom_px]], np.float32)

    M = cv2.getPerspectiveTransform(src,dst)
    binary_warped = cv2.warpPerspective(binary_t, M, binary_t.shape[::-1], flags=cv2.INTER_LINEAR)
    return binary_warped, img, M
#     plt.figure(figsize=(15,15))
#     plt.imshow(binary_warped, cmap='gray')
#     plt.show()
#     print(binary_warped.shape)


def slide_window(binary_warped):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # plt.plot(histogram)
    # plt.show()
    mid = int(binary_warped.shape[1]/2)
    xleft_current = np.argmax(histogram[:mid])
    xright_current = np.argmax(histogram[mid:]) + mid

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    n_window = 9
    window_height = int(binary_warped.shape[0]/n_window)
    margin = 100
    min_pixels = 50

    left_lane_ind = []
    right_lane_ind = []

    for window in range(n_window):
        y_low = binary_warped.shape[0] - (window+1)*window_height
        y_high = binary_warped.shape[0] - (window)*window_height
        x_left_low = xleft_current - margin
        x_left_high = xleft_current + margin
        x_right_low = xright_current - margin
        x_right_high = xright_current + margin

        cv2.rectangle(out_img, (x_left_low,y_low), (x_left_high,y_high), (0,255,0),2)
        cv2.rectangle(out_img, (x_right_low,y_low), (x_right_high,y_high), (0,255,0),2)

        left_box_ind = ((nonzerox >= x_left_low) & (nonzerox < x_left_high) & (nonzeroy < y_high) & (nonzeroy >= y_low)).nonzero()[0]
        right_box_ind = ((nonzerox >= x_right_low) & (nonzerox < x_right_high) & (nonzeroy < y_high) & (nonzeroy >= y_low)).nonzero()[0]
        # print(left_box_ind)
        left_lane_ind.append(left_box_ind)
        right_lane_ind.append(right_box_ind)

        if len(left_box_ind) > min_pixels:
            xleft_current = np.int(np.mean(nonzerox[left_box_ind]))
        if len(right_box_ind) > min_pixels:
            xright_current = np.int(np.mean(nonzerox[right_box_ind]))

    # Copied
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_ind)
    right_lane_inds = np.concatenate(right_lane_ind)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    return out_img, left_fit, right_fit



def looper(binary_warped, left_fit, right_fit, undist, Minv):
    global combined_sobel, sxbinary, prev_pts_left, prev_pts_right, slide_count
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each

    curr_pts_left = leftx.shape[0]
    curr_pts_right = rightx.shape[0]

    if curr_pts_left < (0.85 * prev_pts_left) or curr_pts_right < (0.85 * prev_pts_right):
        _, left_fit, right_fit = slide_window(binary_warped)
        slide_count += 1
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

    prev_pts_left = curr_pts_left
    prev_pts_right = curr_pts_right

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Calculating offset of car from center
    x1 = pts_left[0,650,0]
    x2 = pts_right[0,650,0]
    xc = x1 + (x2 - x1)/2
    center_offset = (binary_warped.shape[1])/2 - xc
    if center_offset < 0:
        offset_str = 'Left'
    elif center_offset > 0:
        offset_str = 'Right'
    else:
        offset_str = 'Center'
    center_offset = abs(center_offset)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Curvature
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    # print(ym_per_pix, xm_per_pix)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    center_curverad = (left_curverad + right_curverad)/2
    # Now our radius of curvature is in meters
    # print("Radius of Curvature = ", center_curverad)
    # print(left_curverad, 'm', right_curverad, 'm')

    cv2.putText(result,'Radius of Curvature: {0:.2f} m'.format(center_curverad),(10,300),2,1.0,[255,255,255],2,cv2.LINE_AA)
    cv2.putText(result,'Vehicle is {0:.2f}m {1} of center'.format(center_offset*xm_per_pix, offset_str),(10,350),4,1.0,[255,255,255],2,cv2.LINE_AA)

    cv2.putText(result,'L-Sobel',(100,20),1,1.0,[255,255,255],1,cv2.LINE_AA)
    cv2.putText(result,'Combined Threshold',(415,20),1,1.0,[255,255,255],1,cv2.LINE_AA)
    cv2.putText(result,'Lane Points',(730,20),1,1.0,[255,255,255],1,cv2.LINE_AA)
    cv2.putText(result,'Lane Polygon',(1045,20),1,1.0,[255,255,255],1,cv2.LINE_AA)

    # Store previous 5 curvature values

    # Merge Plot
    a1 = np.copy(result)
    a2 = cv2.resize(combined_sobel, (0,0), fx=0.25, fy=0.25)
    a2 = np.dstack((a2, a2, a2))*255
    a3 = cv2.resize(sxbinary, (0,0), fx=0.25, fy=0.25)
    a3 = np.dstack((a3, a3, a3))*255
    a4 = cv2.resize(out_img, (0,0), fx=0.25, fy=0.25)
    a5 = cv2.resize(color_warp, (0,0), fx=0.25, fy=0.25)

    a1[50:a2.shape[0]+50,:a2.shape[1],:] = a2
    a1[50:a3.shape[0]+50,a3.shape[1]:2*a3.shape[1],:] = a3
    a1[50:a3.shape[0]+50,2*a4.shape[1]:3*a4.shape[1],:] = a4
    a1[50:a5.shape[0]+50,3*a5.shape[1]:4*a5.shape[1],:] = a5

    # final = np.hstack([a1, np.vstack([a5,a2,a3,a4])])

    # print(result.shape)
    cv2.imshow("Out", a1)
    # cv2.imshow("Lane", color_warp)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    return a1, left_curverad, right_curverad, left_fit, right_fit


cap = cv2.VideoCapture('project_video.mp4')
print(cap.isOpened())

ret, frame = cap.read()
i = 0
lcurve_history = np.zeros(5)
rcurve_history = np.zeros(5)

if ret == True:
    binary_warped, undist, M = get_warped(frame)
    out_img, left_fit, right_fit = slide_window(binary_warped)

m,n = binary_warped.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (n,m))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        binary_warped, undist, M = get_warped(frame)
        result, lcurve, rcurve, left_fit, right_fit = looper(binary_warped, left_fit, right_fit, undist, np.linalg.inv(M))
        out.write(result)

        # Checking curve deviation over last five curvature values
        lcurve_history[i%5] = lcurve
        rcurve_history[i%5] = rcurve

        # if np.std(lcurve_history) > 2000 or np.std(rcurve_history) > 2000:
        #     out_img, left_fit, right_fit = slide_window(binary_warped)

        # i += 1

        # cv2.imshow('frame',result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

print("Video Ended.")
print("Sliding Window called {} times!".format(slide_count))
cap.release()
out.release()
cv2.destroyAllWindows()
