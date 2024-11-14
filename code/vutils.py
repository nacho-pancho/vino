import numpy as np
import skimage.transform as trans
import cv2
import matplotlib.pyplot as plt

def compute_offsets(annotations):
    sync_1 = annotations["sync_1_frame"]
    sync_2 = annotations["sync_2_frame"]

    if sync_1 < 0:
        sync_1 = 0
        print("WARNING: Assuming sync frame 1 is 0 (does not seem right but...)")

    if sync_2 < 0:
        sync_2 = 0
        print("WARNING: Assuming sync frame 2 is 0 (does not seem right but...)")

    if sync_1 == 0 and sync_2 == 0:
        print("WARNING: both sync frames are 0. Did you really annotate this?")

    #  En realidad el offset tiene que hacerse siempre con respecto a la 
    # cámara 1 porque los tiempos anotados son sobre la cámara 1.
    #
    return [0,sync_2-sync_1]

    #if sync_1 < sync_2:
    #    # marker appeared in an earlier frame in camera 1 => it started AFTER camera 2
    #    # so we discard sync_1 - sync_2 frames from camera 2 to put them in sync
    #    offset = [0,sync_2 - sync_1]
    #else:
    #    # vice versa
    #    offset = [sync_1 - sync_2,0]
    print(f"Frame offsets: input 1 {offset[0]} input 2 {offset[1]}")
    return offset

def generate_output_dir(camera_a,camera_b,toma,ini_frame,fin_frame):
    if camera_b is not None:
        return f"{camera_a}+{camera_b}_toma{toma}_frames_{ini_frame}_{fin_frame}"
    else:
        return f"{camera_a}_toma{toma}_frames_{ini_frame}_{fin_frame}"


def generate_annotations_filename(camera_a,camera_b):
    if camera_b is not None:
        return f"{camera_a}+{camera_b}.json"
    else:
        return f"{camera_a}.json"
        

def generate_calibration_directory(camera_a,camera_b):
    if camera_b is not None:
        return f"{camera_a}+{camera_b}.calib"
    else:
        return f"{camera_a}.calib"
        
def fast_rot(img,rot):
    if rot < 0:
        rot += 360
    if rot == 0:
        return img
    elif rot == 90:
        return np.transpose(np.flip(img,axis=0),(1,0,2))
    elif rot == 270:
        return np.flip(np.transpose(img,(1,0,2)),axis=0)
    elif rot == 180:
        return np.flip(np.flip(img,axis=0),axis=1)
    else:
        return (255*trans.rotate(img,rot,resize=True)).astype(np.uint8) # rotation scales colors to 0-1!!


def guess_orientation(cap):
    """
    Guess orientation of video frames
    """
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.25) # start somewhere in the middle of the footage
    frame = None
    angle = None
    blue_profile = None
    n = 0
    while (cap.isOpened()): # ----- loop over frames
        # Capture frame-by-frame
        if frame is None:
            ret, frame = cap.read()
        else:
            ret, frame = cap.read(frame)
        if not ret:
            break
        n += 1
        # first we check whether it is landscape or portrait
        # should be portrait.
        # this is done only once
        h,w,c = frame.shape
        if angle == None:
            if h < w:
                angle = 90
            else:
                angle = 0
            print("height",h,"width",w,"angle",angle)
        #
        # rotate to portrait
        #
        rot_frame = fast_rot(frame,angle)
        #
        # now see where there is more blue
        # should be on the lower half
        if blue_profile is None:
            # frames are BGR
            blue_profile = np.squeeze(np.sum(rot_frame[:,:,0],axis=1))
        else:
            blue_profile += np.squeeze(np.sum(rot_frame[:,:,0],axis=1))
        if n == 100:
            if blue_profile[0] > blue_profile[-1]:
                angle = 270
            #cap.release()
            return angle
        #if not n % 100:
        #    plt.figure()
        #    plt.subplot(1,2,1)
        #    plt.plot(blue_profile*(1/n))
        #    plt.subplot(1,2,2)
        #    print(np.max(rot_frame))
        #    plt.imshow(np.flip(rot_frame,2)*(1/np.max(rot_frame)))
        #    plt.show()
