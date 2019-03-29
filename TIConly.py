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
	print('***************************GETTING THE TIC ONLY*****************************')
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
	testflag = 'yes'; # Activate test mode
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
		print('******************** PLOTTING AND SAVING TIC *****************************');
		from matplotlib import pyplot as plt

		#TIC test function
		params, TIC, fitt = pm.testTIC(imarray, newres, time, type + fit, compressfactor);

		# Plotting function
		# from matplotlib import pyplot as plt
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		# tracker = pm.IndexTracker(ax, imarray[0,30,0,:,:,:])
		# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
		# plt.show()

		# Save avg fit data
		print('Saving Parameters...');print(str(datetime.now()));
		np.savetxt(name + '-' + day + type + 'PARAMS.txt', params, delimiter=',')

		# Save avg fit data
		print('Saving TIC...');print(str(datetime.now()));
		np.savetxt(name + '-' + day + type + 'TIC.txt', TIC, delimiter=',')

		# Save avg fit data
		print('Saving Fit...');print(str(datetime.now()));
		np.savetxt(name + '-' + day + type + 'FITT.txt', fitt, delimiter=',')

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