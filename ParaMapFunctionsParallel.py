from __future__ import print_function
import sys, os, glob
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import scipy as sp 
import scipy.misc
from scipy import stats
from scipy.optimize import curve_fit
import scipy.ndimage as nd
from math import exp, floor, ceil
from time import ctime, sleep
from datetime import datetime
import xml.etree.ElementTree as ET
import dicom
from sklearn.metrics import mean_squared_error
import skimage.transform
from skimage.morphology import opening, disk,closing,ball,erosion, dilation
from skimage.filters import gaussian, threshold_otsu, sobel, rank
from skimage.measure import label, regionprops
from sklearn.metrics import mean_squared_error
import nibabel as nib
from joblib import Parallel, delayed
import multiprocessing
import tempfile
import shutil
import nibabel as nib

#### Should not be used on Sherlock
#from matplotlib import pyplot as plt
#from matplotlib.pyplot import figure, show
#from lmfit import  Model 
#import cv2
#import PIL
#from PIL import Image
#from resizeimage import resizeimage
#import tifffile as tff
#import libtiff
#from pylab import *
#from medpy.io import load, save
#import SimpleITK as sitk
#from skimage.viewer import ImageViewer
#####

# ## Plotting function to check out image at any location of code.
# plotimg = imarray[0,0,0,:,:,:] # give it a 3D, not 6D
# fig = plt.figure();
# ax = fig.add_subplot(111);
# tracker = IndexTracker(ax, plotimg);
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll);
# plt.show();

def prep_img(data, type, format, maskflag, name, day, cut=[[0,0,0,0],[0,0,0,0],[0,0,0,0]]):#[[-1,-1,5,5],[-1,-1,25,5],[-1,-1,5,5]]
	# The better cut >> cut=[[-1,-1,5,5],[-1,-1,25,5],[-1,-1,5,5]]
	print('*************************** ******************** ****************************')
	print('*************************** Prepare Image Volume ****************************')
	print('*************************** ******************** ****************************')

	# 1. Loading DATA: should give 3 outputs: imarray (6D), res, timeconsta
	print('******************************* Load Data ***********************************')
	if format == '3DDICOMs': ### This currently doesn't work and isn't used. User can sen
		imarray, res, time = readDICOM3D(data,0,cut) #Output: imarray and res#
		timeconsta = time/(imarray.shape[1]+1); #Output#
	
	elif format == '4DNIFTI':
		img = nib.load(data);
		data1 = img.get_data();
		data2 = data1.swapaxes(0,3);
		data3 = data2.swapaxes(1,2);
		imarray = data3[np.newaxis,:,np.newaxis,:,:,:]; #Output# Needs to be 1,t,1,z,y,x
		del data1, data2, data3;
		hdr = img.header
		raw = hdr.structarr
		tmpres = raw['pixdim']
		res = np.array([tmpres[1],tmpres[2],tmpres[3]]); #Output# Should be X, Y, Z
		timeconsta = tmpres[4];#Output# AEK - Added Sept 24, 2017: had a bug in previous code - was using a wrong timeconst - this should solve it. 
		del hdr, img;

	elif format == '3DXMLs':
		imarray, res, time = read3D(data,0,cut) #Output: imarray and res#
		timeconsta = time/(imarray.shape[1]+1); #Output#
		
	#org1 image: original without downsampling, masking, adjustments and substractions
	if imarray.shape[1] > 240:
		imarray_org1 = imarray[:,0:240,:,:,:,:];
		imarray = imarray[:,0:240,:,:,:,:];
	else:
		imarray_org1 = imarray[:,:,:,:,:,:];
	print(imarray_org1.shape);

	# 2. Get and apply pre-selected mask
	print('*************************** Load Pre-Selected Mask *********************')
	print('Getting and Applying Pre-Selected MASK:');print(str(datetime.now()));
	if format == '3DXMLs':
		maskdirectory = '/scratch/users/elkaffas/DanDan2/Masks/';
		#maskdirectory = pathonly + '/Masks/'; >> Activate this instead of above
		#dcmdirectory = '/Users/ahmedelkaffas/Documents/GoogleDrive/AEKDocs/Stanford/Project_ParaMapTexture/PythonParametricMaps/Test3DCEUSData/';
		#dcmdirectory = '/mnt/c/Users/AEK/GoogleDrive/AEKDocs/Stanford/Project_ParaMapTexture/PythonParametricMaps/Test3DCEUSData/';
		imarray_org4, mask = pre_mask(imarray_org1,maskdirectory,name,day,'dcm');del imarray_org4;
		# mask = mask[np.newaxis,np.newaxis,np.newaxis,:,:,:];
		# mask2 = resampler(mask, mask.shape, 0, res, newres, 1);
		# mask2.astype('float'); 
		# mask3 = np.squeeze(mask2);
		# mask4 = closing(mask3, ball(8));
		# mask4 = mask; del mask;
		# imarray = np.ma.array(imarray, mask=((imarray+5)*mask4)==0,fill_value=np.nan).filled();

	elif format == '4DNIFTI':
		# Mask naming conventin: name+'-'+day+'MASK.nii.gz' EXAMPLE: C3P3TL-D01-B1MASK.nii.gz
		path = os.path.normpath(data);
		pathonly = os.path.dirname(path);
		print('Getting mask from:');print(pathonly + '/Masks/');
		maskdirectory = pathonly + '/Masks/';
		
		# Messy - but resamples the mask seperate from image - image has to be full when resampling.
		imarray_org4, mask = pre_mask(imarray_org1,maskdirectory,name,day,'nii'); del imarray_org4;
		print('Mask size');print(mask.shape); #COMMENT: mask comes out same dim as imarray_org1
		
	# 3. Box my image based on mask
	print('***************************** Box Image and Mask **************************')
	imarray = np.squeeze(imarray); #6D to 4D
	slice_z, slice_y, slice_x = nd.find_objects(mask>0)[0];
	imarray = imarray[:, slice_z, slice_y, slice_x];
	mask = mask[slice_z, slice_y, slice_x];
	imarray = imarray[np.newaxis,:,np.newaxis,:,:,:]; #Back to 6D
	print('Image/mask size after boxing (6D):',end=' '); print(imarray.shape);print(mask.shape);

	# 4a. Re/Downsample 
	print('***************************** Downsample *********************************')
	# Auto set newres based on min - used for patient data
	autoresFlag = 'yes';
	if autoresFlag == 'yes':
		#del newres;
		if min(res) < 0.3:
			newres = np.array([0.3,0.3,0.3])
		else:
			newres = np.array([0.3,0.3, 0.3])

	# 4b. Do the re/downsampling	
	print('Downsampling from res:',end=' ');print(res,end=' ');print('to newres:',end=' ');print(newres);
	print(str(datetime.now()));
	# Image
	imarray = resampler(imarray, imarray.shape, 0, res, newres, 3);
	imarray = imarray.astype('float');
	# Mask
	mask = mask[np.newaxis,np.newaxis,np.newaxis,:,:,:];
	mask2 = resampler(mask, mask.shape, 0, res, newres, 1); 
	mask3 = np.squeeze(mask2);
	mask4 = closing(mask3, ball(5)); del mask2, mask3, mask; 
	##imarray_org2 = imarray.copy(); #org image: without masking, adjustments and substractions

	# 5. Find flash, set limits, and remove flash frames, and clean updata
	print('*************************** Adjust Sequence *****************************')
	print('Adjust Sequence:');print(str(datetime.now()));
	imarray = adjust_sequence(imarray,type);
	#imarray_org3 = imarray.copy(); #org image: without masking, and substractions

	# 6. Mask image
	print('******************************* Mask Image *****************************')
	imarray = np.squeeze(imarray);
	imarray=imarray+5;
	imarray = np.ma.array(imarray, mask=(imarray*mask4)==0,fill_value=np.nan).filled();
	imarray = imarray[np.newaxis,:,np.newaxis,:,:,:]; #Back to 6D
	imarray = imarray.astype('int');
	print('Image size after masking and resampling (6D):',end=' '); print(imarray.shape);

	# 7. Apply auto mask; doesn't work very well for now - NOT USED 
	print('*************************** Auto Masking *******************************')
	print('CURRENTLY NOT USED')
	if maskflag == 'yes':
		print('Auto-Masking:');print(str(datetime.now()));
		#imarray = imarray - np.mean(imarray[:,0:4,:,:,:,:],axis=1);imarray[imarray < 1]=0;
		imarray, mask = masking(imarray,newres); 

	# 8. Do some substractions to clean artifacts in image
	print('*************************** Clean up the 4D ****************************')
	imarray_tosub = np.nanmean(imarray[:,0:2,:,:,:,:],axis=1);
	imarray = imarray - imarray_tosub; 
	imarray[imarray < 1]=0;
	#imarray_org3b = imarray.copy(); #org image: without masking

	return imarray, res, newres, timeconsta, imarray_org1, mask4; #You can choose org1, 2, 3 or 4 ...

def pre_mask(imarray_org,maskdirectory,name,day,maskformat):
	# Masking should make everything outside of mask np.nan
	## Currentl (01/12/2017) This function just preps the mask. 
	# Read, process and Apply mask to all 4D
	# Mask must be named as 'm904d20150303MASK.dcm' - chrchtr count must be kept the same and format
	#dcmdirectory = '/scratch/users/elkaffas/ParaMap/Masks/'
	#dcmdirectory = '/Users/ahmedelkaffas/Documents/Google Drive/AEK Docs/Stanford/Project_ParaMapTexture/PythonParametricMaps/Test3DCEUSData/'
	
	if maskformat == 'dcm':
		maskfilename = maskdirectory+name+'d'+day+'MASK.dcm';
		#maskfilename = maskdirectory+day+'MASK.dcm';
		print(maskfilename);
		info = dicom.read_file(maskfilename);
		try:
			mask = info.pixel_array;
		except ValueError:
			mask = np.fromstring(info.PixelData,dtype=np.int8);
			mask = mask[:-1];
			mask = mask.reshape((info.NumberofFrames,info.Rows,info.Columns))

	elif maskformat == 'nii':
		maskfilename = maskdirectory+name+'-'+day+'MASK.nii.gz';
		print(maskfilename);
		img = nib.load(maskfilename);
		data1 = img.get_data();
		mask = data1.swapaxes(0,2);
	
	# DOING ALL THIS OUTSIDE IN PREP FUNCTION
	# # Some processing to adjust for mask and image dimmension missmatch. 
	# if imarray_org.shape[3]>mask.shape[0]: 
	# 	tempx = imarray_org.shape[3]; 
	# else: 
	# 	tempx = mask.shape[0];

	# if imarray_org.shape[4]>mask.shape[1]: 
	# 	tempy = imarray_org.shape[4]; 
	# else: 
	# 	tempy = mask.shape[1];

	# if imarray_org.shape[5]>mask.shape[2]: 
	# 	tempz = imarray_org.shape[5]; 
	# else: 
	# 	tempz = mask.shape[2];

	# temp1 = np.zeros(tempx*tempy*tempz).reshape(tempx,tempy,tempz).astype('float64');
	# temp2 = np.zeros(imarray_org.shape[0]*imarray_org.shape[1]*imarray_org.shape[2]*tempx*tempy*tempz).reshape(imarray_org.shape[0],imarray_org.shape[1],imarray_org.shape[2],tempx,tempy,tempz).astype('float64');
	# temp1[:mask.shape[0],:mask.shape[1],:mask.shape[2]]=mask;
	# temp2[:,:,:,:imarray_org.shape[3],:imarray_org.shape[4],:imarray_org.shape[5]]=imarray_org;
	# mask = np.copy(temp1); 
	# #mask[mask == 1] = 0; >> Can be included for MevisLab masks
	# mask[mask < 1] = 0;mask[mask >0] = 1;
	# imarray_org = np.copy(temp2).astype('float'); #imarray_org=np.array(imarray_org*mask);
	# #mask=mask > 0;

	# # We can mask the image here
	# imarray_org=np.squeeze(imarray_org);
	# imarray_org = np.ma.array(imarray_org, mask=((imarray_org)*mask)==0,fill_value=np.nan).filled();
	# imarray_org =imarray_org[np.newaxis,:,np.newaxis,:,:,:];

	return imarray_org, mask; #Image returned unmasked -only minor dim fix if needed. 

def masking(imarray,res,type='conservative',lowcap_std=8):# Originally 14 with std    
	# Masking should make everything outside of mask np.nan
	# New masking by Alireza
	# AEK NOTES: This should not be used as of March 17th, 2017 - major issues and causes errors in data analysis. Needs major fixes/adjustments.

    #type='conservative'
    if type is 'conservative':
        #print('conservative mask')
        ball_size=13;
    else:
        ball_size=5;
        
    # squueze data to reduce dimension from 6d to 4d
    imarray=np.squeeze(imarray).astype('float');
    
    # del imarray;
    
    # look at distribution and detect changes using std
    std=np.std(imarray[5:,:,:,:],0); #std=np.std(np.squeeze(imarray),0);
    #projmax=np.max(imarray[:,:,:,:],0);

    #get binay image using otsu
    mask_raw=std>np.max([lowcap_std, .55*threshold_otsu(std)])
    #print(threshold_otsu(projmax));print(.55*threshold_otsu(projmax));
    #mask_raw=projmax>5;
    #mask_raw=projmax>np.min([lowcap_std, .55*threshold_otsu(projmax)])
    #mask_raw=(std>.55*threshold_otsu(std));

    # remove background noise 
    #mask_original=scipy.ndimage.median_filter(mask_raw, size=(10,10,10)) 
    #mask_original=scipy.ndimage.median_filter(mask_original, size=(5,5,5))
    #del mask_raw;
    mask_original = mask_raw;
    
    # close holes
    #mask_filled=mask_original;
    mask_filled=scipy.ndimage.maximum_filter(mask_original, size=(10,10,10)) 
    mask_filled = closing(mask_filled, ball(ball_size));
    
    # repeat: close holes
    if type is 'conservative':
        mask_filled=scipy.ndimage.maximum_filter(mask_filled, size=(10,10,10))
        mask_filled=scipy.ndimage.median_filter(mask_filled, size=(10,10,10))
           
    # remove remaining background
    x=mask_filled*np.nanmean(imarray,0);    
    mask_filled=(x>.1*threshold_otsu(x))
    del x;

    # remove background noise 
    mask_filled=scipy.ndimage.median_filter(mask_filled, size=(15,15,15)) #was 10
    # mask_filled=scipy.ndimage.median_filter(mask_filled, size=(5,5,5))
    
    # close holes
    mask_filled=scipy.ndimage.maximum_filter(mask_filled, size=(2,2,2))  
     
    imarray = np.ma.array(imarray, mask=imarray*(mask_filled==0),fill_value=np.nan).filled();
    #imarray = np.ma.array(imarray, mask=imarray*mask_filled,fill_value=np.nan).filled();

    imarray =imarray[np.newaxis,:,np.newaxis,:,:,:]
    
    return imarray, mask_filled; #,mask_original

def adjust_sequence(imarray,type):
	# looks for the first position that is x% (in decimal) above the average of the first 3 frames -3
	x=0.005;
	t0 = np.nanmean(imarray[0,0,0,:,:,:])
	t1 = np.nanmean(imarray[0,1,0,:,:,:])
	t2 = np.nanmean(imarray[0,2,0,:,:,:])
	#t3 = imarray[:,3,:,:,:,:].mean()
	tavg = np.nanmean([t0,t1,t2]);
	shapes = imarray.shape; 
	start = 0;
	for i in range(3,shapes[1]):
		tn = np.nanmean(imarray[0,i,0,:,:,:]);
		if tn > tavg*(x+1):
			start = i-3;
			break;
	
	#Check that the start is never negative
	if start<0:
		start = 0;
	print('Frame Start:'); print(start);

    #Figure out which type of cine and adjust by removing flash,etc. 
	if type == 'Infusion':
		imarray = imarray[:,start:,:,:,:,:];
		shapes = imarray.shape;
		print('Remove Flash Frames: Infusion');
		flash = [];
		ref = np.nanmean(imarray[:,0:25,:,:,:,:])
		for k in range(0,(shapes[1]-10)):
			tn = np.nanmean(imarray[:,k,:,:,:,:]);
			if tn > ref*(0.20+1):
				flash.append(k);print(flash);
		flash = np.array([flash]);
		imarray = np.delete(imarray,flash, axis=1);

	elif type == 'Molecular':
		imarray = imarray[:,start:,:,:,:,:];
		shapes = imarray.shape;print(shapes);
		print('Remove Flash Frames: Molecular');
		flash = [];
		ref = np.nanmean(imarray[:,0:25,:,:,:,:])
		for k in range(0,(shapes[1]-10)):
			tn = np.nanmean(imarray[:,k,:,:,:,:])
			if tn > ref*(0.20+1):
				flash.append(k);
		flash = np.array([flash]);print('Flash Frames:');print(flash);
		imarray = np.delete(imarray,flash, axis=1);
		imarray = imarray[:,0:(flash[0,0]+20),:,:,:,:];

	else: #Bolus
		if shapes[1]>175:
			imarray = imarray[:,start:175,:,:,:,:];
			shapes = imarray.shape;
		else:
			imarray = imarray[:,start:,:,:,:,:];

	return imarray;

def resampler(imarray, curshape, newshape, curres, newres, intorder):
	if newshape == 0:
		axial = curshape[4]*curres[1];#y
		lateral = curshape[5]*curres[0];#x
		width = curshape[3]*curres[2];#z
		#return block_reduce(imarray, block_size=(1,1,1,newres/res[2],newres/res[1],newres/res[0]), fund=np.mean)
		imarray2 = np.zeros((1,curshape[1],1,int(width/newres[2]),int(axial/newres[1]),int(lateral/newres[0])), dtype='uint8');
		print('Downsampling Frame:',end=' ');
		for t in xrange(curshape[1]):
			print(t,end=',');
			imarray2[0,t,0,:,:,:] = skimage.transform.resize(imarray[0,t,0,:,:,:], (int(width/newres[2]),int(axial/newres[1]),int(lateral/newres[0])), order=intorder,preserve_range=True, mode='constant', cval=0,); #order 3 is bicubic interpolation
	else:
		imarray2 = np.zeros((curshape[0],curshape[1],curshape[2],newshape[3],newshape[4],newshape[5]));
		for t in xrange(curshape[1]):
			imarray2[0,t,0,:,:,:] = skimage.transform.resize(imarray[0,t,0,:,:,:], (newshape[3],newshape[4],newshape[5]),order=intorder,preserve_range=True, mode='constant', cval=1,);
	return imarray2;

def avgfit(img, res, time, tf, compressfactor):
	global voxelscale, compression, imgshape, timeconst, times, xlist, ylist, zlist, windows, typefit;
	voxelscale = res[0]*res[1]*res[2];
	compression = compressfactor; 
	imgshape = img.shape;
	typefit = tf;

	#1b. Creat time point and position lists
	timeconst = time;#time/(img.shape[1]+1);
	times = np.arange(1,img.shape[1]+1);

	TIC = generate_TIC(img[0,:,0,:,:,:]);

	# Normalize array - should put normalizer in data_fit function... 
	normalizer = np.max(TIC[:,1]);
	TIC[:,1] = TIC[:,1]/normalizer;

	# Bunch of checks
	if np.isnan(np.sum(TIC[:,1])):
		#params = np.empty((5));
		#params[:] = np.nan; maps = generate_maps(params,z,y,x,maps);
		print('STOPPED:NaNs in the VOI')
		return;
	if np.isinf(np.sum(TIC[:,1])):
		#params = np.empty((5));
		#params[:] = np.nan; maps = generate_maps(params,z,y,x,maps);
		print('STOPPED:InFs in the VOI')
		return;

	# Do the fitting
	try:
		params, popt, RMSE = data_fit(TIC,typefit,normalizer);
	except RuntimeError:
		print('RunTimeError')
		#params = np.array([np.max(TIC[:,1])*normalizer, np.max(TIC[:,0])*np.max(TIC[:,1])*normalizer, np.max(TIC[:,0]), np.max(TIC[:,0])*4, 0]);
		params = np.array([np.max(TIC[:,1])*normalizer, np.trapz(TIC[:,1]*normalizer, x=TIC[:,0]), TIC[np.argmax(TIC[:,1]),0], np.max(TIC[:,0])*2, 0]);
		return params;
	
	# # For testing the plot - uncomment matplotlib imports
	# yaj = bolus_lognormal(TIC[:,0], popt[0], popt[1], popt[2], popt[3])
	# plt.plot(TIC[:,0],TIC[:,1],'x',TIC[:,0],yaj,'r-')
	# plt.show()

	print('RMSE:'); print(RMSE);
	# Some post-fitting filters
	# if normalizer < 1:
	# 	params[:] = 0.1;
	# if RMSE > 1:#0.16
	# 	params[:] = 0.1;
	# if params[params<0].any():
	# 	params[:] = 0.1; 

	return params;

def testTIC(img, res, time, tf, compressfactor):
	from matplotlib import pyplot as plt
	global voxelscale, compression, imgshape, timeconst, times, xlist, ylist, zlist, windows, typefit;
	voxelscale = res[0]*res[1]*res[2];
	compression = compressfactor; 
	imgshape = img.shape;
	typefit = tf;

	#1b. Creat time point and position lists
	timeconst = time;#time/(img.shape[1]+1);
	times = np.arange(1,img.shape[1]+1);

	TIC = generate_TIC(img[0,:,0,:,:,:]);

	# Normalize array - should put normalizer in data_fit function... 
	normalizer = np.max(TIC[:,1]);
	TIC[:,1] = TIC[:,1]/normalizer;

	# Bunch of checks
	if np.isnan(np.sum(TIC[:,1])):
		#params = np.empty((5));
		#params[:] = np.nan; maps = generate_maps(params,z,y,x,maps);
		print('STOPPED:NaNs in the VOI')
		return;
	if np.isinf(np.sum(TIC[:,1])):
		#params = np.empty((5));
		#params[:] = np.nan; maps = generate_maps(params,z,y,x,maps);
		print('STOPPED:InFs in the VOI')
		return;

	# Do the fitting
	try:
		params, popt, RMSE = data_fit(TIC,typefit,normalizer);
	except RuntimeError:
		print('RunTimeError: Params Are No Good')
		#params = np.array([np.max(TIC[:,1])*normalizer, np.max(TIC[:,0])*np.max(TIC[:,1])*normalizer, np.max(TIC[:,0]), np.max(TIC[:,0])*4, 0]);
		params = np.array([np.max(TIC[:,1])*normalizer, np.trapz(TIC[:,1]*normalizer, x=TIC[:,0]), TIC[np.argmax(TIC[:,1]),0], np.max(TIC[:,0])*2, 0]);
		return params;
	
	# For testing the plot - uncomment matplotlib imports
	fitt = bolus_lognormal(TIC[:,0], popt[0], popt[1], popt[2], popt[3])
	plt.plot(TIC[:,0],TIC[:,1],'x',TIC[:,0],fitt,'r-')
	plt.show()

	print('RMSE:'); print(RMSE);

	return params, TIC, fitt;

def paramap(img, res, time, tf, compressfactor):
	print('*************************** Starting Parameteric Map *****************************')
	print('Prep For Loop:');print(str(datetime.now()))
	#1a. Windowing and image info
	global windSize, stepSize, voxelscale, compression, imgshape, timeconst, times, xlist, ylist, zlist, windows, typefit;
	windSize = (2,2,2);
	stepSize = (1,1,1);
	voxelscale = res[0]*res[1]*res[2];
	compression = compressfactor; 
	imgshape = img.shape;
	typefit = tf;
	#img = img - np.mean(img[:,0:4,:,:,:,:],axis=1);img[img < 1]=0;

	#1b. Creat time point and position lists
	timeconst = time;#time/(img.shape[1]+1);1
	times = np.arange(1,img.shape[1]+1);
	xlist = np.arange(int(floor(windSize[0]/2)),int(img.shape[5]-floor(windSize[0]/2)),stepSize[0]);
	ylist = np.arange(int(floor(windSize[1]/2)),int(img.shape[4]-floor(windSize[1]/2)),stepSize[1]);
	zlist = np.arange(int(floor(windSize[2]/2)),int(img.shape[3]-floor(windSize[2]/2)),stepSize[2]);

	#1c. Make my windows
	windows = sliding_window(img,(1,1,1,windSize[2],windSize[1],windSize[0]),(1,1,1,stepSize[2],stepSize[1],stepSize[0]), False);
	del img;

	#2. Build array of windowed values
	print('Paraloop start:');print(str(datetime.now()));
	num_cores = 5;#multiprocessing.cpu_count();
	folder = tempfile.mkdtemp(); #Create a tmp file to store process data
	maps_name = os.path.join(folder, 'maps');
	maps = np.zeros((5,1,1,imgshape[3],imgshape[4],imgshape[5])).astype('float64');
	maps = np.memmap(maps_name, dtype=maps.dtype, shape=maps.shape, mode='w+');
	#dump(globmaps, globmaps_name);
	#globmaps = load(globmaps_name, mmap_mode='r');
	#maps = np.mean(np.array(Parallel(n_jobs=num_cores)(delayed(calculate_paramap)(index,xlist,ylist,zlist,imgshape,times, windows, windSize, compression, voxelscale, typefit, timeconst) for index in np.ndindex(xlist.shape[0], ylist.shape[0], zlist.shape[0]))),axis=0);
	Parallel(n_jobs=num_cores)(delayed(calculate_paramap)(maps,index) for index in np.ndindex(xlist.shape[0], ylist.shape[0], zlist.shape[0]));
	shutil.rmtree(folder);

	#3. Sending out resampled map with curr res.
	return maps;

def calculate_paramap(maps,index):
	# Create blank version of array - for final image(s). Based on an input number for x paramteres based on function used.
	# Allows for maximum 5 parametric maps per type of imaging or model.
	# Get indicies
	params = [];
	indexinv = index[::-1]; 
	x = xlist[index[0]];y = ylist[index[1]];z = zlist[index[2]];
	k = index[2];j=index[1];i=index[0];

	#Check that the window is not just nans or junk
	if np.isnan(np.nanmean(windows[0,:,0,k,j,i,0,0,0,:,:,:])):
		return;
	if np.nanmean(windows[0,:,0,k,j,i,0,0,0,:,:,:]) < 0:
		return;

	TICz = generate_TIC(windows[0,:,0,k,j,i,0,0,0,:,:,:]);

	# Normalize array - should put normalizer in data_fit function... 
	normalizer = np.max(TICz[:,1]);
	TICz[:,1] = TICz[:,1]/normalizer;

	# Bunch of checks for nans - dangerous for memory
	if np.isnan(np.sum(TICz[:,1])):
		#params = np.empty((5));
		#params[:] = np.nan; maps = generate_maps(params,z,y,x,maps);
		return;
	if np.isinf(np.sum(TICz[:,1])):
		#params = np.empty((5));
		#params[:] = np.nan; maps = generate_maps(params,z,y,x,maps);
		return;

	# Do the fitting
	try:
		params, popt, RMSE = data_fit(TICz,typefit,normalizer);
	except RuntimeError:
		return;
	
	# Some post-fitting filters
	# if normalizer < 1:
	# 	print('normalizer:'); print(normalizer);
	# 	params[:] = 0.1;
	if RMSE > 0.3:#0.16
		#print('RMSE:'); print(RMSE);
		params[:] = 0.1;
	# if params[params[0:3]<0].any():
	# 	params[0:3] = 0.1; 

	# Generate masks for blending - blending is (old + new)/2 if there is an old, else it's just new
	maps = generate_maps(params,z,y,x,maps);

def generate_TIC(window):
	TICtime=[];TIC=[];
	for t in xrange(0,times.shape[0]):
		TICtime.append(times[t]); 
		tmpwin = window[t,:,:,:];		
		TIC.append(np.around(np.exp(tmpwin[~np.isnan(tmpwin)]/compression).mean()/voxelscale, decimals=1));
	TICz = np.array([TICtime,TIC]).astype('float64'); TICz = TICz.transpose();
	TICz[:,1]=TICz[:,1]-np.mean(TICz[0:2,1]);#Substract noise in TIC before contrast.
	if TICz[np.nan_to_num(TICz)<0].any():#make the smallest number in the TIC 0.
		TICz[:,1]=TICz[:,1]+np.abs(np.min(TICz[:,1]));
	else:
		TICz[:,1]=TICz[:,1]-np.min(TICz[:,1]);
	return TICz;

def generate_maps(params,z,y,x,globmaps):
	# #Generate final para maps
	for p in xrange(params.shape[0]):
		globmaps[p,0,0,z,y,x]=params[p];
	# for p in xrange(params.shape[0]):
	# 	ddd1 = globmaps[p,0,0,(z-int(floor(windSize[2]/2))):(z+int(floor(windSize[2]/2))),(y-int(floor(windSize[1]/2))):(y+int(floor(windSize[1]/2))),(x-int(floor(windSize[0]/2))):(x+int(floor(windSize[0]/2)))]
	# 	ddd2 = np.where(ddd1 > 0,2,1); # masks resulting array (dd2) with 1 where it is 0 and 2 where more -- Blending within the windows instead of space between step sizes. Blending takes average of the two values set to be at the same location > Could be implemented better - maybe take median?
	# 	globmaps[p,0,0,(z-int(floor(windSize[2]/2))):(z+int(floor(windSize[2]/2))),(y-int(floor(windSize[1]/2))):(y+int(floor(windSize[1]/2))),(x-int(floor(windSize[0]/2))):(x+int(floor(windSize[0]/2)))] = (globmaps[p,0,0,(z-int(floor(windSize[2]/2))):(z+int(floor(windSize[2]/2))),(y-int(floor(windSize[1]/2))):(y+int(floor(windSize[1]/2))),(x-int(floor(windSize[0]/2))):(x+int(floor(windSize[0]/2)))]+params[p])/ddd2;
	
	return globmaps;

def data_fit(TIC,model,normalizer):
	#Fitting function
	#Returns the parameters scaled by normalizer
	#Beware - all fitting - minimization is done with data normalized 0 to 1. 
	if model == 'BolusLognormal':
		#kwargs = {"max_nfev":5000}
		popt, pcov = curve_fit(bolus_lognormal, TIC[:,0], TIC[:,1], p0=(1.0,3.0,0.5,0.1),bounds=([0., 0., 0., -1.], [np.inf, np.inf, np.inf, 10.]),method='trf')#p0=(1.0,3.0,0.5,0.1) ,**kwargs
		popt = np.around(popt, decimals=1);
		auc = popt[0]; rauc=normalizer*popt[0]; mu=popt[1]; sigma=popt[2]; t0=popt[3]; mtt=timeconst*np.exp(mu+sigma*sigma/2);
		tp = timeconst*exp(mu-sigma*sigma); wholecurve = bolus_lognormal(TIC[:,0], popt[0], popt[1], popt[2], popt[3]); pe = normalizer*np.max(wholecurve);
		rt0 = timeconst*t0;# + tp;
		
		# Filters to block any absurb numbers based on really bad fits. 
		if tp > 220: tp = 220; #pe = 0.1; rauc = 0.1; rt0 = 0.1; mtt = 0.1;
		if rt0 > 160: rt0 = 160; #pe = 0.1; rauc = 0.1; tp = 0.1; mtt = 0.1;
		if mtt > 2000: mtt = 2000; #pe = 0.1; rauc = 0.1; tp = 0.1; rt0 = 0.1;
		if pe > 1e+07: pe = 1e+07;
		if rauc > 1e+08: rauc = 1e+08;
		params = np.array([pe, rauc, tp, mtt, rt0]);
		# Get error parameters
		residuals = TIC[:,1] - bolus_lognormal(TIC[:,0], popt[0], mu, sigma, t0);
		ss_res = np.sum(residuals[~np.isnan(residuals)]**2);# Residual sum of squares
		ss_tot = np.sum((TIC[:,1]-np.mean(TIC[:,1]))**2);# Total sum of squares
		r_squared = 1 - (ss_res / ss_tot);# R squared
		RMSE = (scipy.sum(residuals[~np.isnan(residuals)]**2)/(residuals[~np.isnan(residuals)].size-2))**0.5;#print('RMSE 1');print(RMSE);# RMSE
		rMSE = mean_squared_error(TIC[:,1], bolus_lognormal(TIC[:,0], popt[0], mu, sigma, t0))**0.5;#print('RMSE 2');print(rMSE);
		return params, popt, RMSE;

def bolus_lognormal(x, auc, mu, sigma, t0):        
    curve_fit=(auc/(2.5066*sigma*(x-t0)))*np.exp(-1*(((np.log(x-t0)-mu)**2)/(2*sigma*sigma))) 
    return np.nan_to_num(curve_fit)
    
def bolus_lagmodel(x, auc, landa, mu, sigma):
    curve_fit=(auc/2)*landa*np.exp(-landa*x+landa*mu+0.5*(landa**2)*(sigma**2))*(1+sp.special.erf( (x-mu-landa*(sigma**2))/(np.sqrt(2*(sigma**2))) ))  
    return np.nan_to_num(curve_fit) 
    
def bolus_gammamodel(x, auc, beta, alpha, t0):
    alpha1=alpha+1
    curve_fit=(auc/((beta** alpha1)*sp.special.gamma(alpha1)))*((x-t0)**(alpha1-1))*np.exp(-(x-t0)/beta)
    return np.nan_to_num(curve_fit)   
    
def bolus_FPTmodel(x, auc, landa, mu, t0):
    curve_fit=auc*(np.exp(landa)/mu)* 0.3989*np.sqrt(landa)*((mu/(x-t0))**1.5)*np.exp(-0.5*landa*((mu/(x-t0))+((x-t0)/mu)))  
    return np.nan_to_num(curve_fit)
    
def bolus_LDRWmodel(x, auc, landa, mu, t0):
    curve_fit=auc*((np.exp(landa))/mu)*np.sqrt((mu/(x-t0))*(landa/6.2832))*np.exp(-0.5*landa*((mu/(x-t0))+((x-t0)/mu)))    
    return np.nan_to_num(curve_fit)

def sliding_window(a,ws,ss = None,flatten = True):
	# '''
	# Return a sliding window over a in any number of dimensions

	# Parameters:
	#     a  - an n-dimensional numpy array
	#     ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
	#          of each dimension of the window
	#     ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
	#          amount to slide the window in each dimension. If not specified, it
	#          defaults to ws.
	#     flatten - if True, all slices are flattened, otherwise, there is an 
	#               extra dimension for each dimension of the input.

	# Returns
	#     an array containing each n-dimensional window from a
	# '''
	if None is ss:
	  	# ss was not provided. the windows will not overlap in any direction.
		ss = ws
	ws = norm_shape(ws)
	ss = norm_shape(ss)

	# convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
	# dimension at once.
	ws = np.array(ws)
	ss = np.array(ss)
	shape = np.array(a.shape)


	# ensure that ws, ss, and a.shape all have the same number of dimensions
	ls = [len(shape),len(ws),len(ss)]
	if 1 != len(set(ls)):
	  raise ValueError(\
	  'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

	# ensure that ws is smaller than a in every dimension
	if np.any(ws > shape):
	  raise ValueError(\
	  'ws cannot be larger than a in any dimension a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

	# how many slices will there be in each dimension?
	newshape = norm_shape(((shape - ws) // ss) + 1)
	# the shape of the strided array will be the number of slices in each dimension
	# plus the shape of the window (tuple addition)
	newshape += norm_shape(ws)
	# the strides tuple will be the array's strides multiplied by step size, plus
	# the array's strides (tuple addition)
	newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
	strided = ast(a,shape = newshape,strides = newstrides)
	if not flatten:
	  return strided

	# Collapse strided so that it has one more dimension than the window.  I.e.,
	# the new array is a flat list of slices.
	meat = len(ws) if ws.shape else 0
	firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
	dim = firstdim + (newshape[-meat:])
	# remove any dimensions with size 1
	dim = filter(lambda i : i != 1,dim)

	return strided.reshape(dim);

def norm_shape(shape):
	# '''
	# Normalize numpy array shapes so they're always expressed as a tuple, 
	# even for one-dimensional shapes.

	# Parameters
	#     shape - an int, or a tuple of ints

	# Returns
	#     a shape tuple
	# '''
	try:
	  i = int(shape)
	  return (i,)
	except TypeError:
	  # shape was not a number
	  pass

	try:
	  t = tuple(shape)
	  return t
	except TypeError:
	  # shape was not iterable
	  pass

	raise TypeError('shape must be an int, or a tuple of ints')

def view4d(imarray2,p,t,c,z,y,x):
	# Display and scroll through stack of 4D along t and z.
	cv2.namedWindow("Original1", cv2.WINDOW_NORMAL)
	imarray3 = cv2.resize(imarray2[p,0,0,0,:,:],(y*1,x*1))
	cv2.imshow("Original1", imarray3)

	# plt.imshow(imarray2[1,1,:,:])
	# plt.show()

	thresholdlevelT = 0
	thresholdlevelZ = 0

	while True:
		# Need an if statement for min max of z and t.
		k = cv2.waitKey(0) & 0xff
		if k == 27: # ESC
			cv2.destroyAllWindows()
			break
		elif k == 1: # downkey
			thresholdlevelZ = (thresholdlevelZ - 1)
			imarray3 = cv2.resize(imarray2[p,thresholdlevelT,0,thresholdlevelZ,:,:],(y*1,x*1))
			cv2.imshow("Original1", imarray3)
			# plt.imshow(imarray3[thresholdlevelT,thresholdlevelZ,:,:])
			# plt.show()
		elif k == 0: # upkey
			thresholdlevelZ = (thresholdlevelZ + 1)
			imarray3 = cv2.resize(imarray2[p,thresholdlevelT,0,thresholdlevelZ,:,:],(y*1,x*1))
			cv2.imshow("Original1", imarray3)
			# plt.imshow(imarray3[thresholdlevelT,thresholdlevelZ,:,:])
			# plt.show()
		elif k == 2: # leftkey
			thresholdlevelT = (thresholdlevelT - 1)
			imarray3 = cv2.resize(imarray2[p,thresholdlevelT,0,thresholdlevelZ,:,:],(y*1,x*1))
			cv2.imshow("Original1", imarray3)
			# plt.imshow(imarray3[thresholdlevelT,thresholdlevelZ,:,:])
			# plt.show()
		elif k == 3: # rightkey	
			thresholdlevelT = (thresholdlevelT + 1)
			imarray3 = cv2.resize(imarray2[p,thresholdlevelT,0,thresholdlevelZ,:,:],(y*1,x*1))
			cv2.imshow("Original1", imarray3)
			# plt.imshow(imarray3[thresholdlevelT,thresholdlevelZ,:,:])
			# plt.show()

def frame_diff(img):
	# imageVec is the input image volume with dimensions time,z,y,x
	imgDiff = np.zeros((1,img.shape[1],1,img.shape[3],img.shape[4],img.shape[5]), dtype='float');
	for i in range(0,img.shape[1]-1):
		# Original
		# qmsk1 = np.mean(img[:,0:1,:,:,:,:],axis=1); #qmsk2[qmsk1 == 0] = 1; 
		# imgDiff[:,i,:,:,:,:] = img[:,i,:,:,:,:]-qmsk1;

		# Diffs
		imgDiff[:,i,:,:,:,:] = img[:,i+1,:,:,:,:]-img[:,i,:,:,:,:];
	return imgDiff;

def read_xmlraw_image_func(filename):    
	# get .raw filename
	filename_raw=filename[0:len(filename)-3]+('0.raw');
	fff = open(filename_raw,'rb')

	# parsing xml file
	tree = ET.parse(filename);
	root = tree.getroot();

	# ADD MAX FRAMES TO LOAD
	numfiles = len(root); # Comment this out if using max num frames on next lines
	# if len(root)>250:
	# 	numfiles = 250;
	# else:
	# 	numfiles = len(root);

	for i in range(0, numfiles):
		if  root[i].tag=='Columns':
		  M=int(root[i].text);
		if  root[i].tag=='Rows':
		  N=int(root[i].text);  
		if  (root[i].find('Geometry') is None) is False:
		  P=int(root[i].find('Geometry').find('Layers').find('Layer').find('RegionLocationMaxz1').text)+1;
		  voxelX=float(root[i].find('Geometry').find('Layers').find('Layer').find('PhysicalDeltaX').text);
		  voxelY=float(root[i].find('Geometry').find('Layers').find('Layer').find('PhysicalDeltaY').text);
		  voxelZ=float(root[i].find('Geometry').find('Layers').find('Layer').find('PhysicalDeltaZ').text);
		  voxel=[voxelX*10, voxelY*10, voxelZ*10]; 
		if  root[i].tag=='AcquisitionDateTime':
		  tval=root[i].text;
		  dateStr=tval[0:4]+'-'+tval[4:6]+'-'+tval[6:8]+' '+tval[8:10]+':'+tval[10:12]+':'+tval[12:len(tval)];
		  time=float(tval[12:len(tval)])+float(tval[10:12])*60+float(tval[8:10])*3600;

	#print(M,N,P,voxel,tval,dateStr,time);
	shapes = (M,N,P);
	x = np.fromfile(fff,dtype=np.uint8)
	img = np.reshape(x, (P,N,M))

	return img, voxel, time, shapes, dateStr

def read3D(data, newres, cut):#=[[-1,-1,5,5],[-1,-1,25,5],[-1,-1,5,5]]):
	# cut=[[10,15,5,5],[50,5,20,4],[15,15,5,5]] #Size reduce with user selected caps
	# cut=[[-1,-1,5,5],[-1,-1,5,5],[-1,-1,5,5]] #Automatic size reduce
	# cut=[[0,0,5,5],[0,0,20,4],[0,0,5,5]] # Keep original size      
	#print(cut,newres);
	N_lines_z_axis_cut=cut[0][0:2] #10,10	
	N_lines_z_axis_cut_limit=cut[0][2:4]#5,5
	N_lines_y_axis_cut=cut[1][0:2]#50,5	 
	N_lines_y_axis_cut_limit=cut[1][2:4]#20,5
	N_lines_x_axis_cut=cut[2][0:2]##[15, 15]	
	N_lines_x_axis_cut_limit=cut[2][2:4]#5,5
	xmldir = data+('/*.xml');
	xmlnamedir = sorted(glob.glob(xmldir));
	
	# ADD MAX FRAMES TO LOAD
	# if len(xmlnamedir)>250:
	# 	xmlnamedir = xmlnamedir[0:250];
	    
	#img, res, timeinitial, shapes, dateStr = read_xmlraw_image_func(xmlnamedir[0])
	#imarray = np.zeros((1,len(xmlnamedir),1,shapes[2],shapes[1],shapes[0]),dtype='uint8')
	imarray = []#np.zeros((len(xmlnamedir),shapes[2],shapes[1],shapes[0]),dtype='uint8')
	timeinitial = -1000; ix=-1;
	#print(xmlnamedir);
	imi_mid, res, timelast, shapes, dateStr = read_xmlraw_image_func(xmlnamedir[np.uint16(len(xmlnamedir)/2)]); 
	imi10, res, timelast, shapes, dateStr = read_xmlraw_image_func(xmlnamedir[10]); 
	imi_mid, res, timelast, shapes, dateStr = read_xmlraw_image_func(xmlnamedir[1]); 

	for xmlname in xmlnamedir:
		#imarray[0,xmlnamedir.index(xmlname),0,:,:,:], res, timelast, shapes, dateStr = read_xmlraw_image_func(xmlname)
		imi, res, timelast, shapes, dateStr = read_xmlraw_image_func(xmlname); 
		sz=imi.shape

		if timeinitial==-1000 and np.sum(cut) != 0: 			
			timeinitial = timelast
			mp=np.mean(imi10,0)+np.mean(imi_mid,0);
			mpb=mp>0.3*threshold_otsu(mp); 
			lc=rank.otsu(mp.astype('uint16'), disk(10)); mpb2=mp>0.7*np.mean(lc); 
			mpb=(mpb+mpb2)>0;
			mp1=np.sum(mpb[:,40:mp.shape[1]-20],1);
			ix = np.where(mp1>0)
			mp0=np.sum(mpb,0)
			ix0 = np.where(mp0>0)          
			mpz=np.mean(imi10,1)+np.mean(imi_mid,1)+np.mean(imi,1);
			mpzb=mpz>1.2*threshold_otsu(mpz)           
			mpz1=np.sum(mpzb,1)
			iz = np.where(mpz1>0)     
			#ix[0][-1]=sz[1];ix[0][0]=0; #for test
			#ix0[0][-1]=sz[2];ix0[0][0]=0; #for test          
			#iz[0][-1]=sz[0];iz[0][0]=0; #for test   

			if N_lines_y_axis_cut[0]!=0: 
				N_lines_y_axis_cut[0]=max(0,max(N_lines_y_axis_cut[0]*(ix[0][0]>0),ix[0][0]-N_lines_y_axis_cut_limit[0])); 
			if N_lines_y_axis_cut[1]!=0:          
			   	N_lines_y_axis_cut[1]=sz[1]-min(sz[1],max((sz[1]-N_lines_y_axis_cut[1])*(ix[0][-1]<sz[1]),ix[0][-1]+N_lines_y_axis_cut_limit[1]))

			if N_lines_x_axis_cut[0]!=0:          
			   	N_lines_x_axis_cut[0]=max(0,max(N_lines_x_axis_cut[0]*(ix0[0][0]>0),ix0[0][0]-N_lines_x_axis_cut_limit[0])) 
			if N_lines_x_axis_cut[1]!=0:          
			   	N_lines_x_axis_cut[1]=sz[2]-min(sz[2],max((sz[2]-N_lines_x_axis_cut[1])*(ix0[0][-1]<sz[2]),ix0[0][-1]+N_lines_x_axis_cut_limit[1])) 

			if N_lines_z_axis_cut[0]!=0: 
			   	N_lines_z_axis_cut[0]=max(0,max(N_lines_z_axis_cut[0]*(iz[0][0]>0),iz[0][0]-N_lines_z_axis_cut_limit[0])) 
			if N_lines_z_axis_cut[1]!=0:          
			   	N_lines_z_axis_cut[1]=sz[0]-min(sz[0],max((sz[0]-N_lines_z_axis_cut[1])*(iz[0][-1]<sz[1]),iz[0][-1]+N_lines_z_axis_cut_limit[1]))

			#print(N_lines_z_axis_cut,N_lines_y_axis_cut,N_lines_x_axis_cut)

			print('Orginal image volume size',imi.shape)

		if np.sum(cut) == 0:
			print('Orginal image volume size - no change',imi.shape);
			imarray.append(imi)
		
		else:
		    # reduce matrix size      
			imi=imi[N_lines_z_axis_cut[0]:sz[0]-N_lines_z_axis_cut[1],N_lines_y_axis_cut[0]:sz[1]- N_lines_y_axis_cut[1],N_lines_x_axis_cut[0]:sz[2]-N_lines_x_axis_cut[1]]
			
		    # resampling - mostly not used
			if newres!=0:			   
			   imi=resampler_4d(imi, 0, res, newres)
			imarray.append(imi)

	time = timelast - timeinitial; #total time of cine in seconds.
	if newres!=0:
	    print('voxel size is changed from ', res, 'to voxel size of ', newres)
	
	print('Reduced image volume size',imi.shape)

	sh1_og=np.shape(imi_mid);sh1_og=(sh1_og[0]*sh1_og[1]*sh1_og[2])/(1024**2)
	sh1=np.shape(imi);sh1=(sh1[0]*sh1[1]*sh1[2])/(1024**2)
	#print('Reduction rate: ',np.round(100*sh1/sh1_og),'% of original size-- from ',np.round(sh1_og), ' Mbytes to ', np.round(sh1), ' Mbytes')  
	imarray1 = np.zeros((1,len(xmlnamedir),1,imi.shape[0],imi.shape[1],imi.shape[2]),dtype='uint8')
	imarray1[0,:,0,:,:,:] = np.asarray(imarray)
	#imarray=imarray1

	return imarray1, res, time;

def readDICOM3D(data, newres, cut):
    dcmdir = data+('/*.dcm');
    dcmnamedir = sorted(glob.glob(dcmdir));print(dcmnamedir);
    imarray = [];#np.zeros((len(xmlnamedir),shapes[2],shapes[1],shapes[0]),dtype='uint8')
    info = dicom.read_file(dcmnamedir[1])
    tval = info.AcquisitionDateTime
    dateStr=tval[0:4]+'-'+tval[4:6]+'-'+tval[6:8]+' '+tval[8:10]+':'+tval[10:12]+':'+tval[12:len(tval)];
    time_initial=float(tval[12:len(tval)])+float(tval[10:12])*60+float(tval[8:10])*3600;


    for dcmname in dcmnamedir:
        info = dicom.read_file(dcmname)
        zres = info.SpacingBetweenSlices;
        yres = info.PixelSpacing[0];
        xres = info.PixelSpacing[1];
        tval = info.AcquisitionDateTime
        dateStr=tval[0:4]+'-'+tval[4:6]+'-'+tval[6:8]+' '+tval[8:10]+':'+tval[10:12]+':'+tval[12:len(tval)];
        time=float(tval[12:len(tval)])+float(tval[10:12])*60+float(tval[8:10])*3600;

        im = info.pixel_array;
        res=np.array([xres,yres,zres])
        #t = info.NumberOfTemporalPositions; time =0;
        # x = im.asarray().shape[2]
        # y = im.asarray().shape[1]
        # z = im.asarray().shape[0]

        #Shape image into typical mevislab/itk format
        imarray.append(im);

    imarray1 = np.zeros((1,len(dcmnamedir),1,im.shape[0],im.shape[1],im.shape[2]),dtype='uint8')
    imarray1[0,:,0,:,:,:] = np.asarray(imarray)
    fulltime = time - time_initial;
    return imarray1, res, fulltime;

## 3D Plotting >> Edited by AEK
class IndexTracker(object):
    def __init__(self, ax, X):
    	self.ax = ax;
    	ax.set_title('use scroll wheel to navigate images');
    	self.X = X;
    	rows, cols, self.slices = X.shape;
    	self.ind = self.slices//2;
    	self.im = ax.imshow(self.X[:, :, self.ind]);
    	self.update();
    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step));
        if event.button == 'up':
            self.ind = np.clip(self.ind + 1, 0, self.slices - 1);
        else:
            self.ind = np.clip(self.ind - 1, 0, self.slices - 1);
        self.update();
    def update(self):
    	self.im.set_data(self.X[:, :, self.ind]);
    	self.ax.set_ylabel('slice %s' % self.ind);
    	self.im.axes.figure.canvas.draw();