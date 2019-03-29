from __future__ import print_function
import ParaMapFunctionsParallel as pm
import sys, os, glob
import nibabel as nib
from datetime import datetime
import numpy as np
#from matplotlib import pyplot as plt
#import xlrd
#import sys, os
#import SimpleITK as sitk

#Example:
#python /home/elkaffas/bin/RunBolusParaMap.py EPIC7 3DXMLs /scratch/users/elkaffas/ParaMap/LSBolusAV/m267/20150319104853.535_boulous

if __name__ == "__main__":
	# CODE ACTIVATOR
	print('Started:');print(sys.argv[3]);print(str(datetime.now()));
	print('Note that a -Masks- directory must exist as a subdirectory containing masks.')
	print('This version of the code has only been tested with 4DNIFTI')
	path = os.path.normpath(sys.argv[3]);
	splitpath = path.split(os.sep);
	pathonly = os.path.dirname(path)

	# Check which system we're dealing with
	if sys.argv[1] == 'EPIC7':
		compressfactor = 24.09; #24.09; #42.98
	elif sys.argv[1] == 'iU22':
		compressfactor = 42.98;
	else:
		print('TERMINATED: First argument can only be EPIC7 or iU22');
		sys.exit(0);

	# Check which data format we are dealing with
	if sys.argv[2] == '4DNIFTI':
		fullname = splitpath[-1];
		name = fullname[0:6];print(name);
		day = fullname[7:13];print(day);
		format = '4DNIFTI'
	elif sys.argv[2] == '3DXMLs':
		name = splitpath[-2];
		day = splitpath[-1];day = day[0:8];
		format = '3DXMLs'
	else:
		print('TERMINATED: Second argument can only be 4DNIFTI or 3DXMLs');
		sys.exit(0);

	# These will be future options in code input
	testflag = 'no'; # Activate test mode
	maskflag = 'no' # Flag to indicate auto-masking yes/no >> barely used here
	type = 'Bolus'
	fit = 'Lognormal'
	if type == 'Bolus':
		parameters = ['PE','AUC','TP','MTT','T0']

	# Read data	
	imarray, orgres, newres, time, imarray_org, mask = pm.prep_img(sys.argv[3],type,format, maskflag, name, day);
	print('Done 3D to 4D:');print(str(datetime.now()));

	# This is condition to test or look at something specific in code and stop running. 
	# Can copy and past elesewhere. 
	if testflag == 'yes':
		from matplotlib import pyplot as plt

		#TIC test function
		pm.testTIC(imarray, newres, time, type + fit, compressfactor);

		# Plotting function
		from matplotlib import pyplot as plt
		fig = plt.figure()
		ax = fig.add_subplot(111)
		tracker = pm.IndexTracker(ax, imarray[0,30,0,:,:,:])
		fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
		plt.show()

		# Exit right after testing
		sys.exit(0);

	# Run avg fitting
	print('Start Avg Fit:');print(str(datetime.now()))
	params = pm.avgfit(imarray, newres, time, type + fit, compressfactor);
	print('Done Avg Fit:');print(str(datetime.now()));
	print(params);

	# Save avg fit data
	print('Saving Avg Fits...');print(str(datetime.now()));
	np.savetxt(name + '-' + day + type + '.txt', params, delimiter=',')

	print('Saving 4D w/o mask:');print(str(datetime.now()));
	#imarray_swap = np.reshape(imarray_org[0,:,0,:,:,:],(imarray_org.shape[1], imarray_org.shape[3], imarray_org.shape[4],imarray_org.shape[5]));
	imarray_org2 = np.squeeze(imarray_org);
	imarray_org2 = imarray_org2.swapaxes(0,3);imarray_org2 = imarray_org2.swapaxes(1,2);
	affine = np.eye(4)
	niiarray = nib.Nifti1Image(imarray_org2.astype('uint8'),affine);
	niiarray.header['pixdim'] = [4.,orgres[0], orgres[1], orgres[2], time, 0., 0., 0.];
	#niiarray.header['slice_duration'] = time;
	nib.save(niiarray, (name + '-' + day + type + 'Full4D.nii.gz'));
	del imarray_org, imarray_org2;

	# Save the 4D image with mask
	print('Saving 4D w/ mask:');print(str(datetime.now()));
	imarray2 = np.squeeze(imarray);
	imarray2 = imarray2.swapaxes(0,3);imarray2 = imarray2.swapaxes(1,2);
	affine = np.eye(4)
	niiarray = nib.Nifti1Image(imarray2.astype('uint8'),affine);
	niiarray.header['pixdim'] = [4.,newres[0], newres[1], newres[2], time, 0., 0., 0.];
	#niiarray.header['slice_duration'] = time;
	nib.save(niiarray, (name + '-' + day + type + 'Full4DMasked.nii.gz'));
	del imarray2;

	## Save 3D mask image - This is the downsampled and adjusted mask
	print('Saving mask:');print(str(datetime.now()));
	affine = np.eye(4);
	niiarray = nib.Nifti1Image(np.squeeze(mask.swapaxes(0,2).astype('uint8')),affine);#NEED TO FLIP THESE IN X,Y,Z,T for quick mevis read. 
	niiarray.header['pixdim'] = [3., newres[0], newres[1], newres[2], 0., 0., 0., 0.];
	nib.save(niiarray, (name + '-' + day + type + 'FinMask.nii.gz')); 
	del mask;

	# Create linearized 4D
	imarray_lin=np.array(imarray);
	imarray_lin[imarray_lin==0]= -100*compressfactor;
	imarray_lin[np.isnan(imarray_lin)]= -1000000*compressfactor;
	imarray_lin = np.exp(imarray_lin/compressfactor); # Linearized imarray - used for projections and diff.

	## Save 3D MIP image - Max
	print('Saving MIP:');print(str(datetime.now()));
	MIP = np.max(imarray_lin[:,:,:,:,:,:],axis=1); MIP = MIP[np.newaxis,:,:,:,:,:];
	#MIP = MIP - np.mean(imarray_lin[:,0:4,:,:,:,:],axis=1); MIP[MIP < 1]=0;
	affine = np.eye(4);
	niiarray = nib.Nifti1Image(MIP[0,0,0,:,:,:].swapaxes(0,2).astype('float64'),affine);#NEED TO FLIP THESE IN X,Y,Z,T for quick mevis read. 
	niiarray.header['pixdim'] = [3., newres[0], newres[1], newres[2], 0., 0., 0., 0.];
	nib.save(niiarray, (name + '-' + day + type + 'MIP.nii.gz')); 

	## Save 3D AIP image - Avg over first hundred frames... 
	print('Saving AIP:');print(str(datetime.now()));
	AIP = np.mean(imarray_lin[:,:100,:,:,:,:],axis=1); AIP = AIP[np.newaxis,:,:,:,:,:];
	#AIP = AIP - np.mean(imarray_lin[:,0:4,:,:,:,:],axis=1); AIP[AIP < 1]=0;
	affine = np.eye(4);
	niiarray = nib.Nifti1Image(AIP[0,0,0,:,:,:].swapaxes(0,2).astype('float64'),affine);#NEED TO FLIP THESE IN X,Y,Z,T for quick mevis read. 
	niiarray.header['pixdim'] = [3., newres[0], newres[1], newres[2], 0., 0., 0., 0.];
	nib.save(niiarray, (name + '-' + day + type + 'AIP.nii.gz'));

	## Save 3D SIP image - Std 
	print('Saving SIP:');print(str(datetime.now()));
	SIP = np.std(imarray_lin[:,:,:,:,:,:],axis=1); SIP = SIP[np.newaxis,:,:,:,:,:];
	affine = np.eye(4);
	niiarray = nib.Nifti1Image(SIP[0,0,0,:,:,:].swapaxes(0,2).astype('float64'),affine);#NEED TO FLIP THESE IN X,Y,Z,T for quick mevis read. 
	niiarray.header['pixdim'] = [3., newres[0], newres[1], newres[2], 0., 0., 0., 0.];
	nib.save(niiarray, (name + '-' + day + type + 'SIP.nii.gz'));

	## Save 3D SumDiff image - Sum of Differences over first 80 frames
	print('Saving SumDiff:');print(str(datetime.now()));
	diffs = pm.frame_diff(imarray_lin[:,:80,:,:,:,:]); 
	SumDiff = np.sum(diffs,axis=1); 
	SumDiff = SumDiff[np.newaxis,:,:,:,:,:];#SumDiff[SumDiff < 1]=0;
	#SumDiff = SumDiff - np.mean(SumDiff[:,0:4,:,:,:,:],axis=1); SumDiff[SumDiff < 1]=0;
	affine = np.eye(4);
	niiarray = nib.Nifti1Image(SumDiff[0,0,0,:,:,:].swapaxes(0,2).astype('float64'),affine);#NEED TO FLIP THESE IN X,Y,Z,T for quick mevis read. 
	niiarray.header['pixdim'] = [3., newres[0], newres[1], newres[2], 0., 0., 0., 0.];
	nib.save(niiarray, (name + '-' + day + type + 'SumDiff.nii.gz'));

	## Save 3D StdDiff image - Std of Differences over first 80 frames
	print('Saving StdDiff:');print(str(datetime.now()));
	#diffs = pm.frame_diff(imarray);
	StdDiff = np.std(diffs,axis=1); 
	StdDiff = StdDiff[np.newaxis,:,:,:,:,:];#StdDiff[StdDiff < 1]=0;
	#StdDiff = StdDiff - np.mean(StdDiff[:,0:4,:,:,:,:],axis=1); StdDiff[StdDiff < 1]=0;
	affine = np.eye(4);
	niiarray = nib.Nifti1Image(StdDiff[0,0,0,:,:,:].swapaxes(0,2).astype('float64'),affine);#NEED TO FLIP THESE IN X,Y,Z,T for quick mevis read. 
	niiarray.header['pixdim'] = [3., newres[0], newres[1], newres[2], 0., 0., 0., 0.];
	nib.save(niiarray, (name + '-' + day + type + 'StdDiff.nii.gz'));

	## Save 3D MaxDiff image - Max of Differences over first 80 frames
	print('Saving MaxDiff:');print(str(datetime.now()));
	#diffs = pm.frame_diff(imarray);
	MaxDiff = np.max(diffs,axis=1); 
	MaxDiff = MaxDiff[np.newaxis,:,:,:,:,:]; #MaxDiff[MaxDiff < 1]=0;
	#MaxDiff = MaxDiff - np.mean(MaxDiff[:,0:4,:,:,:,:],axis=1); MaxDiff[MaxDiff < 1]=0;
	affine = np.eye(4);
	niiarray = nib.Nifti1Image(MaxDiff[0,0,0,:,:,:].swapaxes(0,2).astype('float64'),affine);#NEED TO FLIP THESE IN X,Y,Z,T for quick mevis read. 
	niiarray.header['pixdim'] = [3., newres[0], newres[1], newres[2], 0., 0., 0., 0.];
	nib.save(niiarray, (name + '-' + day + type + 'MaxDiff.nii.gz'));

	## Save 3D MaxDiff image - Min of Differences over first 80 frames
	print('Saving MinDiff:');print(str(datetime.now()));
	#diffs = pm.frame_diff(imarray);
	MinDiff = np.min(diffs,axis=1); 
	MinDiff = MinDiff[np.newaxis,:,:,:,:,:]; #MaxDiff[MaxDiff < 1]=0;
	#MaxDiff = MaxDiff - np.mean(MaxDiff[:,0:4,:,:,:,:],axis=1); MaxDiff[MaxDiff < 1]=0;
	affine = np.eye(4);
	niiarray = nib.Nifti1Image(MinDiff[0,0,0,:,:,:].swapaxes(0,2).astype('float64'),affine);#NEED TO FLIP THESE IN X,Y,Z,T for quick mevis read. 
	niiarray.header['pixdim'] = [3., newres[0], newres[1], newres[2], 0., 0., 0., 0.];
	nib.save(niiarray, (name + '-' + day + type + 'MinDiff.nii.gz'));

	## Save 3D AvgDiff image - Avg of Differences over first 40 frames
	print('Saving AvgDiff:');print(str(datetime.now()));
	diffs = pm.frame_diff(imarray_lin[:,:40,:,:,:,:]); 
	AvgDiff = np.mean(diffs,axis=1); 
	AvgDiff = AvgDiff[np.newaxis,:,:,:,:,:];#AvgDiff[MaxDiff < 1]=0;
	#MaxDiff = MaxDiff - np.mean(MaxDiff[:,0:4,:,:,:,:],axis=1); MaxDiff[MaxDiff < 1]=0;
	affine = np.eye(4);
	niiarray = nib.Nifti1Image(AvgDiff[0,0,0,:,:,:].swapaxes(0,2).astype('float64'),affine);#NEED TO FLIP THESE IN X,Y,Z,T for quick mevis read. 
	niiarray.header['pixdim'] = [3., newres[0], newres[1], newres[2], 0., 0., 0., 0.];
	nib.save(niiarray, (name + '-' + day + type + 'AvgDiff.nii.gz'));

	## Clear memory by deleting non-necessary images after saving them. 
	del AvgDiff, MinDiff, MaxDiff, StdDiff, SumDiff, SIP, imarray_lin, MIP, AIP;

	# Run para map gen
	print('Start Maps:');print(str(datetime.now()))
	maps = pm.paramap(imarray, newres, time, type + fit, compressfactor);
	#maps = maps.astype('float64');
	print('Done Maps:');print(str(datetime.now()));print('Saving...');

	# Save parametric maps
	for p in xrange(len(parameters)):
		printname = name + '-' + day + type + parameters[p] + '.nii.gz';# FIX NAME SAVING METHOD
		#ParaMapFunctions.view4d(maps,p,0,0,maps.shape[3],maps.shape[4],maps.shape[5]);
		#sitk.WriteImage(sitk.GetImageFromArray(maps[p,0,0,:,:,:]), ('A'+printname)); 
		affine = np.eye(4)
		niiarray = nib.Nifti1Image(maps[p,0,0,:,:,:].swapaxes(0,2).astype('float64'),affine);#NEED TO FLIP THESE IN X,Y,Z,T for quick mevis read. 
		niiarray.header['pixdim'] = [3., newres[0], newres[1], newres[2], 0., 0., 0., 0.];
		nib.save(niiarray, (printname));
	del maps;
	print('Done Maps');print(str(datetime.now()));

	print('Done All');print(str(datetime.now()));