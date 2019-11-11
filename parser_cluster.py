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
        self.frameNum = id
    def append_pt(self, x, y):
        self.xs = np.append(self.xs, x)
        self.ys = np.append(self.ys, y)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: parseTLV.py inputFile.bin")
        sys.exit()

    fileName = sys.argv[1]
    rawDataFile = open(fileName, "rb")
    rawData = rawDataFile.read()
    rawDataFile.close()


# Parse rawData into framesList
framesList = []
while rawData:
    # Seek to the next frame
    offset = rawData.find(MAGIC)
    rawData = rawData[offset:]

    # Make sure there is still enough rawData left to parse
    if len(rawData) < frames.FRAME_HEADER_BYTES:
        break

    header = FrameHeader(*struct.unpack('Q7I', rawData[:FRAME_HEADER_BYTES]))
    rawData = rawData[FRAME_HEADER_BYTES:]
    frame = Frame(header.frameNum)
    for i in range(header.numTLVs):
        tlv_header = TLVHeader(*struct.unpack('2I', rawData[:TLV_HEADER_BYTES]))
        rawData = rawData[TLV_HEADER_BYTES:]
        if tlv_header.type == DETECTED_OBJECTS:
            contents = DetectedObjects(rawData[:tlv_header.length])
            for content in contents.objects:
                frame.append_pt(content.X, content.Y)
        rawData = rawData[tlv_header.length:]
    framesList.append(frame)
        
# perform clustering
for i in range(len(framesList) - 3):

    framebunch = framesList[i:i+3]
    x_axis = np.append(framebunch[0].xs, np.append(framebunch[1].xs, framebunch[2].xs))
    y_axis = np.append(framebunch[0].ys, np.append(framebunch[1].ys, framebunch[2].ys))
    
    if len(x_axis) == 0:
        wq = 1 
        # print("not enough data")
    else:
        df = pd.DataFrame(list(zip(x_axis, y_axis)), columns =['X', 'Y'])
        scaler = StandardScaler() 
        df_scaled = scaler.fit_transform(df) 
        db_default = DBSCAN(eps = 0.1, min_samples = 3).fit(df_scaled)
        labels = db_default.labels_ 
        indices = db_default.core_sample_indices_.astype(int)
        if any(i != -1 for i in labels):
            # print("x-axis:",x_axis)
            # print("y-axis:",y_axis)
            # print("labels:", labels)
            # print("indices:", db_default.core_sample_indices_)
            # TODO: Should be splitting this up by label, this takes all inliers
            print(" {} mean: {},\t{}".format(framebunch[0].frameNum, mean(x_axis[indices]), mean(y_axis[indices])))

        
                    
