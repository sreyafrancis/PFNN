"""
statistics for footsetp & phase generation (preprocess_footstep_pahse.py)
Author: Xiaofeng
"""

import sys
import os
import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters
import ctypes

sys.path.append('./motion')

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots


# Constants from the Windows API
STD_OUTPUT_HANDLE = -11
FOREGROUND_RED    = 0x0004 # text color contains red.
def get_csbi_attributes(handle):
    # Based on IPython's winconsole.py, written by Alexander Belchenko
    import struct
    csbi = ctypes.create_string_buffer(22)
    res = ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, csbi)
    assert res
    (bufx, bufy, curx, cury, wattr,
    left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
    return wattr

data_terrain = [
    './data/animations/LocomotionFlat01_000.bvh',
    './data/animations/LocomotionFlat02_000.bvh',
    './data/animations/LocomotionFlat02_001.bvh',
    './data/animations/LocomotionFlat03_000.bvh',
    './data/animations/LocomotionFlat04_000.bvh',
    './data/animations/LocomotionFlat05_000.bvh',
    './data/animations/LocomotionFlat06_000.bvh',
    './data/animations/LocomotionFlat06_001.bvh',
    './data/animations/LocomotionFlat07_000.bvh',
    './data/animations/LocomotionFlat08_000.bvh',
    './data/animations/LocomotionFlat08_001.bvh',
    './data/animations/LocomotionFlat09_000.bvh',
    './data/animations/LocomotionFlat10_000.bvh',
    './data/animations/LocomotionFlat11_000.bvh',
    './data/animations/LocomotionFlat12_000.bvh',

    './data/animations/LocomotionFlat01_000_mirror.bvh',
    './data/animations/LocomotionFlat02_000_mirror.bvh',
    './data/animations/LocomotionFlat02_001_mirror.bvh',
    './data/animations/LocomotionFlat03_000_mirror.bvh',
    './data/animations/LocomotionFlat04_000_mirror.bvh',
    './data/animations/LocomotionFlat05_000_mirror.bvh',
    './data/animations/LocomotionFlat06_000_mirror.bvh',
    './data/animations/LocomotionFlat06_001_mirror.bvh',
    './data/animations/LocomotionFlat07_000_mirror.bvh',
    './data/animations/LocomotionFlat08_000_mirror.bvh',
    './data/animations/LocomotionFlat08_001_mirror.bvh',
    './data/animations/LocomotionFlat09_000_mirror.bvh',
    './data/animations/LocomotionFlat10_000_mirror.bvh',
    './data/animations/LocomotionFlat11_000_mirror.bvh',
    './data/animations/LocomotionFlat12_000_mirror.bvh',

    './data/animations/WalkingUpSteps01_000.bvh',
    './data/animations/WalkingUpSteps02_000.bvh',
    './data/animations/WalkingUpSteps03_000.bvh',
    './data/animations/WalkingUpSteps04_000.bvh',
    './data/animations/WalkingUpSteps04_001.bvh',
    './data/animations/WalkingUpSteps05_000.bvh',
    './data/animations/WalkingUpSteps06_000.bvh',
    './data/animations/WalkingUpSteps07_000.bvh',
    './data/animations/WalkingUpSteps08_000.bvh',
    './data/animations/WalkingUpSteps09_000.bvh',
    './data/animations/WalkingUpSteps10_000.bvh',
    './data/animations/WalkingUpSteps11_000.bvh',
    './data/animations/WalkingUpSteps12_000.bvh',

    './data/animations/WalkingUpSteps01_000_mirror.bvh',
    './data/animations/WalkingUpSteps02_000_mirror.bvh',
    './data/animations/WalkingUpSteps03_000_mirror.bvh',
    './data/animations/WalkingUpSteps04_000_mirror.bvh',
    './data/animations/WalkingUpSteps04_001_mirror.bvh',
    './data/animations/WalkingUpSteps05_000_mirror.bvh',
    './data/animations/WalkingUpSteps06_000_mirror.bvh',
    './data/animations/WalkingUpSteps07_000_mirror.bvh',
    './data/animations/WalkingUpSteps08_000_mirror.bvh',
    './data/animations/WalkingUpSteps09_000_mirror.bvh',
    './data/animations/WalkingUpSteps10_000_mirror.bvh',
    './data/animations/WalkingUpSteps11_000_mirror.bvh',
    './data/animations/WalkingUpSteps12_000_mirror.bvh',

    './data/animations/NewCaptures01_000.bvh',
    './data/animations/NewCaptures02_000.bvh',
    './data/animations/NewCaptures03_000.bvh',
    './data/animations/NewCaptures03_001.bvh',
    './data/animations/NewCaptures03_002.bvh',
    './data/animations/NewCaptures04_000.bvh',
    './data/animations/NewCaptures05_000.bvh',
    './data/animations/NewCaptures07_000.bvh',
    './data/animations/NewCaptures08_000.bvh',
    './data/animations/NewCaptures09_000.bvh',
    './data/animations/NewCaptures10_000.bvh',
    './data/animations/NewCaptures11_000.bvh',

    './data/animations/NewCaptures01_000_mirror.bvh',
    './data/animations/NewCaptures02_000_mirror.bvh',
    './data/animations/NewCaptures03_000_mirror.bvh',
    './data/animations/NewCaptures03_001_mirror.bvh',
    './data/animations/NewCaptures03_002_mirror.bvh',
    './data/animations/NewCaptures04_000_mirror.bvh',
    './data/animations/NewCaptures05_000_mirror.bvh',
    './data/animations/NewCaptures07_000_mirror.bvh',
    './data/animations/NewCaptures08_000_mirror.bvh',
    './data/animations/NewCaptures09_000_mirror.bvh',
    './data/animations/NewCaptures10_000_mirror.bvh',
    './data/animations/NewCaptures11_000_mirror.bvh',
]
"""
data_terrain = [
    #'./data/animations/LocomotionFlat01_000.bvh',
    #'./data/animations/LocomotionFlat02_000.bvh',
	#'./data/animations/NewCaptures01_000.bvh',
	#'./data/animations/LocomotionFlat02_001.bvh',
	
	'./data/animations/LocomotionFlat06_000.bvh',
    #'./data/animations/LocomotionFlat06_001.bvh',
    #'./data/animations/LocomotionFlat07_000.bvh',
    #'./data/animations/LocomotionFlat08_000.bvh',
]
"""
phase_error = 0
frame_total = 0

acc_step_error = 0
acc_step_count_error = 0
acc_neg_step_count_error = 0
step_total = 0

for data in data_terrain:

	#phases
	f0 = data.replace('.bvh','.phase')
	f1 = data.replace('.bvh','_2.phase')
	
	phase0 = np.loadtxt(f0)
	phase1= np.loadtxt(f1)
	
	frame_count0 = len(phase0)
	frame_count1 = len(phase1)
	if frame_count0 != frame_count1:
		print("%s %d : %s %d" % (f0, frame_count0, f1, frame_count1))
		raise ("invalid input.")

	err = 0
	for i in range(0,frame_count0,1):
		err += np.abs(phase0[i]-phase1[i])
		
	phase_error += err
	frame_total += frame_count0
	
	
	#steps
	f0 = data.replace('.bvh','_footsteps.txt')
	f1 = data.replace('.bvh','_footsteps_2.txt')
	
	with open(f0, 'r') as _f0:
		footsteps = _f0.readlines()
	
	steps0 = [[0 for x in range(2)] for y in range(len(footsteps))]		
	for i in range(len(footsteps)-1):
		step = footsteps[i].split(' ')
		if len(step) >= 2:
			steps0[i][0] = int(step[0])
			steps0[i][1] = int(step[1])
		
	with open(f1, 'r') as _f1:
		footsteps = _f1.readlines()
	steps1 = [[0 for x in range(2)] for y in range(len(footsteps))]
	for i in range(len(footsteps)-1):
		step = footsteps[i].split(' ')
		if len(step) >= 2:
			steps1[i][0] = int(step[0])
			steps1[i][1] = int(step[1])
		
	step_count0 = len(steps0)
	step_count1 = len(steps1)
	step_total += step_count0

	steperror = 0
	stepcount_positive = 0
	stepcount_negative = 0
	
	# if step_count1-step_count0 >= 0:
		# stepcount_positive = step_count1-step_count0
	# else:
		# stepcount_negative = step_count1-step_count0
	stepcounterror = 0#stepcounterror = abs(step_count1-step_count0)
	
	b0 = 0
	b1 = 0
	miss0 = 0
	miss1 = 0
	threshold = 25
	#threshold = max(min(int(0.00222 * frame_count0), 40),5)
	for i in range(0,min(step_count0, step_count1),1):
		if i+b0 < len(steps0) and i + b1 < len(steps1):
			#print("%d:[%d,%d] %d:[%d,%d]" % (i+b0, steps0[i+b0][0], steps0[i+b0][1], i+b1, steps1[i+b1][0], steps1[i+b1][1]))
			if (abs(steps0[i+b0][0]-steps1[i+b1][0]) <= threshold and abs(steps0[i+b0][1]-steps1[i+b1][1]) <= threshold) \
			or (abs(steps0[i+b0][0]-steps1[i+b1][0]) + abs(steps0[i+b0][1]-steps1[i+b1][1]))/2 <= threshold:
				steperror += abs(steps0[i+b0][0] - steps1[i+b1][0])
				steperror += abs(steps0[i+b0][1] - steps1[i+b1][1])
				miss0 = 0
				miss1 = 0
				#print("%f %f %f" %(abs(steps0[i+b0][0] - steps1[i+b1][0]),abs(steps0[i+b0][1] - steps1[i+b1][1]), steperror))
				#print("YES")
			else:
				#print("[%f,%f] : [%f,%f]" % (steps0[i+b0][0], steps0[i+b0][1], steps1[i+b1][0], steps1[i+b1][1]))
				#print("%d, %d == [%f,%f] : [%f,%f]" % (miss0, miss1, steps0[i+b0][0], steps0[i+b0][1], steps1[i+b1][0], steps1[i+b1][1]))
				if miss0 != miss1 or miss0 == 0:
					stepcounterror = stepcounterror+1
					#print("%d:[%d,%d] %d:[%d,%d]" % (i+b0, steps0[i+b0][0], steps0[i+b0][1], i+b1, steps1[i+b1][0], steps1[i+b1][1]))
					#print("NO")
					if steps0[i+b0][0]-steps1[i+b1][0] >= 25 or steps0[i+b0][1]-steps1[i+b1][1] >= 25:
						stepcount_positive += 1
						#print("+[%f,%f] : [%f,%f]" % (steps0[i+b0][0], steps0[i+b0][1], steps1[i+b1][0], steps1[i+b1][1]))
						# if miss0 != miss1 -1:
							# print("[%f,%f] : [%f,%f]" % (steps0[i+b0][0], steps0[i+b0][1], steps1[i+b1][0], steps1[i+b1][1]))
					else:
						if miss0 != miss1 - 1:
							#print("-[%f,%f] : [%f,%f]" % (steps0[i+b0][0], steps0[i+b0][1], steps1[i+b1][0], steps1[i+b1][1]))
							#print("%d, %d == [%f,%f] : [%f,%f]" % (miss0, miss1, steps0[i+b0][0], steps0[i+b0][1], steps1[i+b1][0], steps1[i+b1][1]))
							stepcount_negative -= 1
					
				if steps0[i+b0][0]-steps1[i+b1][0] >= 25 or steps0[i+b0][1]-steps1[i+b1][1] >= 25:
					#b1 = b1 + 1
					b0 = b0 - 1
					miss1 = miss1 + 1
				else:
					#b0 = b0 + 1
					b1 = b1 - 1
					miss0 = miss0 + 1

	#red color
	console_handle = 0
	if stepcount_negative != 0:
		console_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
		reset = get_csbi_attributes(console_handle)
		ctypes.windll.kernel32.SetConsoleTextAttribute(console_handle, FOREGROUND_RED)
	
	print("%s phase error: %f step error: %f, stepcount error: %f, +stepcount %f, -stepcount %f "\
	%(f0, err/frame_count0, steperror/frame_count0, stepcounterror/ step_count0, stepcount_positive/step_count0, stepcount_negative/step_count0))
	
	#end red color
	if stepcount_negative != 0:
		ctypes.windll.kernel32.SetConsoleTextAttribute(console_handle, reset)
	
	acc_step_count_error += stepcounterror
	acc_step_error += steperror
	acc_neg_step_count_error += stepcount_negative
	
total = len(data_terrain)

print("total original steps: %f" %(step_total))
print("acc phase error: %f" %(phase_error/frame_total))
print("acc step error: %f" %(acc_step_error/frame_total))
print("acc step count error: %f" %(acc_step_count_error/step_total))
print("acc negative step count error: %f" %(abs(acc_neg_step_count_error/step_total)))

