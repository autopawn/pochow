import argparse

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
from cv2.saliency import StaticSaliencyFineGrained_create

# Evaluate an image on the given coordinates (appyling interpolation)
def evaluate_on_coords(img,xs,ys,interp=True):
    if not interp:
        xs_round = (np.round(xs)).astype(np.int)
        ys_round = (np.round(ys)).astype(np.int)

        return img[ys_round,xs_round]

    else:
        xs_floor = (np.floor(xs)).astype(np.int)
        ys_floor = (np.floor(ys)).astype(np.int)

        xs_prop = xs-xs_floor
        ys_prop = ys-ys_floor

        img_00 = img[ys_floor  ,xs_floor  ]
        img_01 = img[ys_floor  ,xs_floor+1]
        img_10 = img[ys_floor+1,xs_floor  ]
        img_11 = img[ys_floor+1,xs_floor+1]

        prop_00 = (1-ys_prop) * (1-xs_prop)
        prop_01 = (1-ys_prop) * xs_prop
        prop_10 = ys_prop     * (1-xs_prop)
        prop_11 = ys_prop     * xs_prop

        na = np.newaxis

        fimg = img_00*prop_00[:,:,na] + img_01*prop_01[:,:,na] + img_10*prop_10[:,:,na] + img_11*prop_11[:,:,na]

        fimg = fimg.astype(np.uint8)
        return fimg

# Uses the heat equation to diffuse the saliency evenly in the image
def blur_with_heat_equation(u,x,y,tmax=400,pixres=5000):
    uu = u.astype(np.float)+0.01
    yy = uu*y  # y-isity
    xx = uu*x  # x-isity

    # Resize image
    y2x = uu.shape[0]/uu.shape[1]
    nsy = int((pixres*y2x)**0.5+1)
    nsx = int((pixres/y2x)**0.5+1)

    uu = cv2.resize(uu, dsize=(nsx,nsy), interpolation=cv2.INTER_CUBIC)
    yy = cv2.resize(yy, dsize=(nsx,nsy), interpolation=cv2.INTER_CUBIC)
    xx = cv2.resize(xx, dsize=(nsx,nsy), interpolation=cv2.INTER_CUBIC)

    factor = 0.2

    t = 0
    while t<tmax:

        step = min(tmax-t,factor)

        u_y = uu[:-1,:] - uu[1:,:]
        u_x = uu[:,:-1] - uu[:,1:]

        trasnf_u_s = +np.maximum(u_y,0)*step
        trasnf_u_n = -np.minimum(u_y,0)*step
        trasnf_u_e = +np.maximum(u_x,0)*step
        trasnf_u_w = -np.minimum(u_x,0)*step

        trasnf_prop_s = trasnf_u_s / uu[:-1,:]
        trasnf_prop_n = trasnf_u_n / uu[1:,:]
        trasnf_prop_e = trasnf_u_e / uu[:,:-1]
        trasnf_prop_w = trasnf_u_w / uu[:,1:]

        trasnf_x_s = trasnf_prop_s * xx[:-1,:]
        trasnf_x_n = trasnf_prop_n * xx[1:,:]
        trasnf_x_e = trasnf_prop_e * xx[:,:-1]
        trasnf_x_w = trasnf_prop_w * xx[:,1:]

        trasnf_y_s = trasnf_prop_s * yy[:-1,:]
        trasnf_y_n = trasnf_prop_n * yy[1:,:]
        trasnf_y_e = trasnf_prop_e * yy[:,:-1]
        trasnf_y_w = trasnf_prop_w * yy[:,1:]

        uu[:-1,:] += -trasnf_u_s + trasnf_u_n
        uu[1:,:]  += -trasnf_u_n + trasnf_u_s
        uu[:,:-1] += -trasnf_u_e + trasnf_u_w
        uu[:,1:]  += -trasnf_u_w + trasnf_u_e

        xx[:-1,:] += -trasnf_x_s + trasnf_x_n
        xx[1:,:]  += -trasnf_x_n + trasnf_x_s
        xx[:,:-1] += -trasnf_x_e + trasnf_x_w
        xx[:,1:]  += -trasnf_x_w + trasnf_x_e

        yy[:-1,:] += -trasnf_y_s + trasnf_y_n
        yy[1:,:]  += -trasnf_y_n + trasnf_y_s
        yy[:,:-1] += -trasnf_y_e + trasnf_y_w
        yy[:,1:]  += -trasnf_y_w + trasnf_y_e

        t += factor

    # Back to coordinates
    yy /= uu
    xx /= uu

    uu = cv2.resize(uu, dsize=(u.shape[1],u.shape[0]), interpolation=cv2.INTER_CUBIC)
    yy = cv2.resize(yy, dsize=(u.shape[1],u.shape[0]), interpolation=cv2.INTER_CUBIC)
    xx = cv2.resize(xx, dsize=(u.shape[1],u.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Retrieve final distributions
    return (uu,xx,yy)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--input", required=True, help="path to input image")
ap.add_argument("-o","--output", required=True, help="path to output image")
ap.add_argument('-s',"--spectral", action='store_true',help="Use spectral residual for saliency.")
ap.add_argument('-p',"--plot", action='store_true',help="Plot results.")
ap.add_argument('-p2',"--plot2", action='store_true',help="Plot results with extra details.")
ap.add_argument('-t',"--time", type=int, default=400, help='Difussion time')

args = vars(ap.parse_args())

# Load the input image
image = cv2.imread(args["input"])

# Compute saliency map

if args["spectral"]:
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    saliency_method = "spectral residual"
else:
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    saliency_method = "fine grained"

(success, ss) = saliency.computeSaliency(image)

# Blur saliency along with x-isity and y-isity

xx,yy = np.meshgrid(np.arange(ss.shape[1]),np.arange(ss.shape[0]))
(ssf,xxf,yyf) = blur_with_heat_equation(ss,xx,yy,tmax=args["time"])

# Evaluate image on the resulting coordinates
imagef = evaluate_on_coords(image,xxf,yyf)

# Save output image
cv2.imwrite(args["output"],imagef)

# Plot if required:
if args["plot"] or args["plot2"]:
    n_cols = 4 if args["plot2"] else 2

    fig, axs = plt.subplots(2,n_cols,sharex=True,sharey=True)

    fig.suptitle('Content-aware scaling: '+args["input"])

    axs[0,0].set_title("Input image")
    axs[0,0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

    axs[0,1].set_title("Initial saliency $U$ (%s)"%saliency_method)
    axs[0,1].imshow(ss,vmin=0)

    axs[1,0].set_title("Output image")
    axs[1,0].imshow(cv2.cvtColor(imagef,cv2.COLOR_BGR2RGB))

    axs[1,1].set_title("Saliency $U$ at $t=%d$"%args["time"])
    axs[1,1].imshow(ssf,vmin=0,vmax=np.max(ss))

    if args["plot2"]:
        axs[0,2].set_title("Initial $\\frac{\hat{X}}{U}$")
        axs[0,2].imshow(xx)

        axs[0,3].set_title("Initial $\\frac{\hat{Y}}{U}$")
        axs[0,3].imshow(yy)

        axs[1,2].set_title("$\\frac{\hat{X}}{U}$ at $t=%d$"%args["time"])
        axs[1,2].imshow(xxf)

        axs[1,3].set_title("$\\frac{\hat{Y}}{U}$ at $t=%d$"%args["time"])
        axs[1,3].imshow(yyf)

    plt.show()

