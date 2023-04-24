import numpy as np
import cv2
import open3d as o3d


def rearrange(spoint):
    a = spoint[spoint[:,1].argsort()]
    a1 = a[0:2,:]
    a2 = a[2:4,:]
    a1 = a1[a1[:,0].argsort()]
    a2 = a2[a2[:,0].argsort()[::-1]]
    return np.concatenate((a1, a2), axis=0)


def getedge(zn,xn,originzn,originxn):
    A = np.concatenate((zn,-xn),axis=1)[0:2,0:2]
    B = (originxn - originzn)[0:2,0:2]
    if np.linalg.det(A) == 0:
        A = np.concatenate((zn,-xn),axis=1)[1:3,1:3]
        B = (originxn - originzn)[1:3,1:3]
    Ainv = np.linalg.inv(A)
    x = Ainv @ B #optimized
    return originzn + x[0]*zn


def getSquare(plane, d, shape_origin):
    origin_shape = np.array([shape_origin[0], (-plane[3]-plane[0]*shape_origin[0]-plane[2]*shape_origin[2])/plane[1], shape_origin[2]])
    origin = origin_shape.reshape((-1,1))
    #origin = np.array([0,-plane[3]/plane[1],0]).reshape((-1,1))
    zvec = np.array([shape_origin[0],0,-(plane[3]+plane[0]*shape_origin[0])/plane[2]]).reshape((-1,1)) - origin
    xvec = np.array([-(plane[3]+plane[2]*shape_origin[2])/plane[0],0,shape_origin[2]]).reshape((-1,1)) - origin
    zn = (zvec/np.linalg.norm(zvec)) #test non square
    xn = (xvec/np.linalg.norm(xvec))
    originx1 = (origin + d*xn).reshape((-1,1)) #1
    originx2 = (origin - d*xn).reshape((-1,1)) #3
    originz1 = (origin + d*zn).reshape((-1,1)) #2
    originz2 = (origin - d*zn).reshape((-1,1)) #4
    #x1&z1
    p1 = getedge(zn,xn,originx1,originz1)
    p2 = getedge(zn,xn,originx2,originz1)
    p3 = getedge(zn,xn,originx2,originz2)
    p4 = getedge(zn,xn,originx1,originz2)
    points = np.concatenate((p1,p2,p3,p4),axis=1)
    #print(points)
    return points


def getPlane(rgbd, intrinsic):
    pc = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcl = o3d.t.geometry.PointCloud.to_legacy(pc)
    downpcl = pcl.voxel_down_sample(voxel_size=0.04)
    plane_model, _ = downpcl.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=3000)
                                        #or 0.05?
    #inlier_cloud = downpcl.select_by_index(inliers)
    [a, b, c, d] = plane_model
    #ref_square = inlier_cloud.get_center()
    ref_square = [0, -0.0, -4] #bypass

    square = getSquare(plane_model,1,ref_square) #1.5
    point2d = cv2.projectPoints(square,(3.14,0,0),(0,0,0),intrinsic,0)[0].reshape(4,2).astype(np.int32)

    #dst test
    # m = 0.5
    # W = 1000; H = 1000; r = 50
    # dst_ori = np.array([W/2 + r*ref_square[0], H + r*ref_square[2]])
    # print(dst_ori)
    # #dst = np.float32([[500,500],[500+100*m,500],[500+100*m,500+100*m],[500,500+100*m]]) #this need to be adjust according to y and z
    # dst = np.float32([dst_ori+np.array([-m*50,-m*50]), dst_ori+np.array([m*50,-m*50]), dst_ori+np.array([m*50,m*50]), dst_ori+np.array([-m*50,m*50])])

    return point2d



def getTransform(point2d,image):

    src = np.float32(rearrange(point2d).tolist())
    # print('------------')
    # print(point2d)
    # print(src)
    # print('------------')
    m = 1.2
    dst = np.float32([[200,300],[200+100*m,300],[200+100*m,300+100*m],[200,300+100*m]])
    h = cv2.getPerspectiveTransform(src, dst)
    wrapped = cv2.warpPerspective(image, h, (500,500))
    mask = cv2.drawContours(image, [point2d], -1, 255, -1)


    cv2.putText(img=mask, text='1', org=src[0].astype(np.int32), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
    cv2.putText(img=mask, text='2', org=src[1].astype(np.int32), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
    cv2.putText(img=mask, text='3', org=src[2].astype(np.int32), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
    cv2.putText(img=mask, text='4', org=src[3].astype(np.int32), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
    
    return [wrapped, mask]