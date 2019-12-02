import struct
import sys
import math
from collections import namedtuple
from statistics import mean
import frames
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
# TODO 1: (NOW FIXED) Find the first occurrence of magic and start from there
# TODO 2: Warn if we cannot parse a specific section and try to recover
# TODO 3: Remove error at end of file if we have only fragment of TLV

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
# Byte count constants
FRAME_HEADER_BYTES = 36
TLV_HEADER_BYTES = 8
OBJECTS_HEADER_BYTES = 4
OBJECT_BYTES = 12
STATS_BYTES = 24


# TLV type code constants
DETECTED_OBJECTS = 1
RANGE_PROFILE = 2
STATS = 6


FrameHeader = namedtuple(
    'FrameHeader',
    'magic version length platform frameNum cpuCycles numObj numTLVs')
TLVHeader = namedtuple(
    'TLVHeader',
    'type length')

class DetectedObjects(object):
    """Encapsulates a DetectedObjects type TLV."""

    class DetectedObject(object):
        def __init__(self, objTuple, xyzQFormat):
            try:
                self.RangeIdx = objTuple[0]
                self.DopplerIdx = objTuple[1]
                self.PeakVal = objTuple[2]
                self.X = objTuple[3]*1.0/(1 << xyzQFormat)
                self.Y = objTuple[4]*1.0/(1 << xyzQFormat)
                self.Z = objTuple[5]*1.0/(1 << xyzQFormat)
                self.Range = math.sqrt(pow(self.X, 2) + pow(self.Y, 2))
            except Exception as e:
                raise ParseError(
                    'Could not parse object with tuple {}: {}'.format(
                        objTuple, e))

    def __init__(self, data):
        self.objects = []
        vals = struct.unpack('2H', data[:OBJECTS_HEADER_BYTES])
        self.numDetectedObj = vals[0]
        self.xyzQFormat = vals[1]

        data = data[OBJECTS_HEADER_BYTES:]
        for i in range(self.numDetectedObj):
            objTuple = struct.unpack('3H3h', data[:OBJECT_BYTES])
            obj = self.DetectedObject(objTuple, self.xyzQFormat)
            self.objects.append(obj)
            data = data[OBJECT_BYTES:]

        if len(data) != 0:
            # TLV type classes are supposed to receive exactly as much data
            # as their header indicates is contained in the TLV content.
            # There should be no data left in the buffer after they have parsed
            # all the content.
            raise ParseError(
                'Received more data than expected. '
                'Indicates earlier parsing error.')

    def __str__(self):
        result = "\tDetect Obj:\t%d\n"%(self.numDetectedObj) 
        for i in range(self.numDetectedObj):
            obj = self.objects[i]
            result += "\tObjId:\t%d\n "%(i)
            result += "\t\tDopplerIdx:\t%d\n"%(obj.DopplerIdx)
            result += "\t\tRangeIdx:\t%d\n"%(obj.RangeIdx)
            result += "\t\tPeakVal:\t%d\n"%(obj.PeakVal)
            result += "\t\tX:\t\t%07.3f\n"%(obj.X)
            result += "\t\tY:\t\t%07.3f\n"%(obj.Y)
            result += "\t\tZ:\t\t%07.3f\n"%(obj.Z)
            result += "\t\tRange:\t\t%07.3fm\n"%(obj.Range)
        return result


class Frame(object):
    def __init__(self, id):
        self.xs = np.array([])
        self.ys = np.array([])
        self.zs = np.array([])
        self.frameNum = id
    def append_pt(self, x, y, z):
        self.xs = np.append(self.xs, x)
        self.ys = np.append(self.ys, y)
        self.zs = np.append(self.zs, z)

def evaluate(fileName, X_list, Y_list):
    x_axis = []
    y_axis = []
    f = open(fileName, "r")
    p = f.readline()
    position = int(p)
    print('The position is : ',position)
    for line in f:
        l = line.split()
        x_axis.append(float(l[1]))
        y_axis.append(float(l[2]))
    #print("list of x values are: ",x_axis)
    #print("list of y values are: ",y_axis)
    X = []
    Y = []
    while position<len(X_list):
        #print("position: ", position)
        X.append(X_list[position])
        Y.append(Y_list[position])
        position = position + 4
        

    #print("\n\ntaking only one X val per second: ",X)
    #print("\n\ntaking only one Y val per second: ",Y)

    X_difference = []
    Y_difference = []
    #skipping the first 8 data points(first 5 are no data and the other 3 are for testing)
    i=8
    while i<len(X):
        if(X[i] == 'Not enough data' or X[i] == 'No cluster'):
            i=i+1
        else:
            X_diff = abs(X[i] - x_axis[i])
            Y_diff = abs(Y[i] - y_axis[i])
            X_difference.append(X_diff)
            Y_difference.append(Y_diff)
            i=i+1
    print("\n\nThe difference in X_axis is: ",X_difference)
    print("\n\nThe difference in Y_axis is: ",Y_difference)
    print("\nThe mean of X_difference is: ",mean(X_difference))
    print("\n The mean of Y_difference is: ",mean(Y_difference))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: parseTLV.py inputFile.bin")
        sys.exit()

    fileName = sys.argv[1]  
    ground_truth_fileName = sys.argv[2]
    #ground_truth_fileName = 'labeling\\third_trial.txt'
    rawDataFile = open(fileName, "rb")
    rawData = rawDataFile.read()
    rawDataFile.close()


# Parse rawData into framesList
framesList = []
frameCount = 0
while rawData:
    # Seek to the next frame
    offset = rawData.find(MAGIC)
    rawData = rawData[offset:]

    # Make sure there is still enough rawData left to parse
    frameCount = frameCount + 1
    if (len(rawData) < frames.FRAME_HEADER_BYTES or len(rawData) < 500):
        break

    header = FrameHeader(*struct.unpack('Q7I', rawData[:FRAME_HEADER_BYTES]))
    rawData = rawData[FRAME_HEADER_BYTES:]
    #frame = Frame(header.frameNum)
    frame = Frame(frameCount)
    for i in range(header.numTLVs):
        tlv_header = TLVHeader(*struct.unpack('2I', rawData[:TLV_HEADER_BYTES]))
        rawData = rawData[TLV_HEADER_BYTES:]
        if tlv_header.type == DETECTED_OBJECTS:
            contents = DetectedObjects(rawData[:tlv_header.length])
            for content in contents.objects:
                frame.append_pt(content.X, content.Y, content.Z)
        rawData = rawData[tlv_header.length:]
    framesList.append(frame)
    

frameCombination = 5
NumberOfClusterPoints = np.array([])
object1 = np.array([-1,-1])
framecomboArray = np.array([-1,-1,-1,-1])
index = 0
time = 0
frameCombinationNumber = 0
X_list = []
Y_list = []
# perform clustering
for i in range(1,len(framesList) - frameCombination,frameCombination):

    framebunch = framesList[i:i+frameCombination]
    index = index + 1
    x_axis = np.array([])
    y_axis = np.array([])
    z_axis = np.array([])
    for j in range(frameCombination):
        x_axis = np.append(x_axis, framebunch[j].xs)
        y_axis = np.append(y_axis, framebunch[j].ys)
        z_axis = np.append(z_axis, framebunch[j].zs)
    
    #print("Number of points in framecombination ",len(x_axis))
    #print("x-axis: ",x_axis) 
    #print("y-axis: ",y_axis)
    #print("z-axis: ",z_axis)    

    if len(x_axis) < 2:
        wq = 1 
        #framecomboArray = np.vstack([framecomboArray,[-1,-1,-1,-1]])
        #print("Not Enough Data")
        print("{} {} Not enough data ".format(time,frameCombinationNumber))
        X_list.append('Not enough data')
        Y_list.append('Not enough data')
    else:
        df = pd.DataFrame(list(zip(x_axis, y_axis)), columns =['X', 'Y'])
        
        scaler = StandardScaler() 
        df_scaled = scaler.fit_transform(df) 
        db_default = DBSCAN(eps = 0.15, min_samples = 3).fit(df_scaled)
        labels = db_default.labels_ 
        indices = db_default.core_sample_indices_.astype(int)
        core_samples_mask = np.zeros_like(db_default.labels_, dtype=bool)
        core_samples_mask[db_default.core_sample_indices_] = True
        uniqueLabels = set(labels)
        #print(indices)
        #print(labels)
        maxNumPoints = 0
        clusterNum = -1
        
        if any(j != -1 for j in labels):
            for k in uniqueLabels:
                if(k!=-1):
                    temp = sum(j==k for j in labels)
                    if(temp>maxNumPoints):
                        maxNumPoints = temp
                        clusterNum = k
            NumberOfClusterPoints = np.append(NumberOfClusterPoints,maxNumPoints)
            class_member_mask = (labels == clusterNum)
            X = x_axis[class_member_mask & core_samples_mask]
            Y = y_axis[class_member_mask & core_samples_mask]
            Z = z_axis[class_member_mask & core_samples_mask]
            #print(" {} mean: {},\t{}, {}".format(framebunch[0].frameNum, mean(X), mean(Y), maxNumPoints))

            print("{} {} {} {}".format(time,frameCombinationNumber,mean(X),mean(Y)))
            X_list.append(mean(X))
            Y_list.append(mean(Y))
            object1 = np.vstack([object1,[mean(X),mean(Y)]])
            framecomboArray = np.vstack([framecomboArray,[int(index),mean(X), mean(Y),maxNumPoints]])
        else:
            print("{} {} No cluster found ".format(time,frameCombinationNumber))
            X_list.append('No cluster')
            Y_list.append('No cluster')
    if(frameCombinationNumber == 3):
        time = time + 1
        frameCombinationNumber = 0
    else:
        frameCombinationNumber = frameCombinationNumber + 1

i = 1
while i < len(framecomboArray): 
	#print(str(int(framecomboArray[i][0])))
	startIndex = framecomboArray[i][0]
	X_start = framecomboArray[j][1]
	Y_start = framecomboArray[j][2]
	MaxClusterSize = framecomboArray[j][3] 
	j = i + 1
	while(j < len(framecomboArray) and (framecomboArray[j][0] - framecomboArray[j-1][0] <=2) ):
		MaxClusterSize = MaxClusterSize + int(framecomboArray[j][3])
		j = j+1
	if((j-i) > 1):
		meanClusterSize = MaxClusterSize/(j-i)
		X_end = framecomboArray[j-1][1]
		Y_end = framecomboArray[j-1][2]
		if(meanClusterSize > 10):
				print(" Human Detected cluster Size "+str(meanClusterSize))
		elif(abs(X_end-X_start)<0.5 and abs(Y_end-Y_start)<0.5):
				print("Human Detected No motion")
		else:
			print("Rodent detected cluster size = "+str(meanClusterSize))
	i = j 
evaluate(ground_truth_fileName, X_list, Y_list)
