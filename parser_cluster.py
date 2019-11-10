import struct
import sys
import math
import frames
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler 
# TODO 1: (NOW FIXED) Find the first occurrence of magic and start from there
# TODO 2: Warn if we cannot parse a specific section and try to recover
# TODO 3: Remove error at end of file if we have only fragment of TLV

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: parseTLV.py inputFile.bin")
        sys.exit()

    fileName = sys.argv[1]
    rawDataFile = open(fileName, "rb")
    rawData = rawDataFile.read()
    rawDataFile.close()

    framesList = []
    while rawData:
        #print("inside rawData")
        # Seek to the next frame
        offset = rawData.find(MAGIC)
        rawData = rawData[offset:]

        # Make sure there is still enough data left to parse
        if len(rawData) < frames.FRAME_HEADER_BYTES:
            break

        # Read raw data into Frame object
        frame = frames.Frame()
        rawData = frame.ParseHeader(rawData)
        rawData = frame.ParseTLVs(rawData)
        framesList.append(frame)

    #for frame in framesList:
        #print("printing frame")
        #print("the header of frame: ",frame.header)
        #print(frame)
        
    for frame in framesList:
        print("printing frame ")
        #print(frame)
        x_axis = []
        y_axis = []       
        #print("the number of detected objects: ", frame.header.numObj)
        for tlv in frame.tlvs:
            #print("header type:", tlv.header.type)
            if tlv.header.type == 1:
                print("the number of detected objects are: ",tlv.contents.numDetectedObj)
                for i in range(tlv.contents.numDetectedObj):
                    #print("X:",tlv.contents.objects[i].X)
                    #print("Y:",tlv.contents.objects[i].Y)
                    #print("Z:",tlv.contents.objects[i].Z)
                    x_axis.append(tlv.contents.objects[i].X)
                    y_axis.append(tlv.contents.objects[i].Y)
    
        print("x-axis:",x_axis)
        print("y-axis:",y_axis)
        plt.scatter(x_axis,y_axis)
        #plt.show()
        if len(x_axis) == 0:
            print("not enough data")
        else:
            df = pd.DataFrame(list(zip(x_axis, y_axis)), 
                       columns =['X', 'Y'])
            scaler = StandardScaler() 
            df_scaled = scaler.fit_transform(df) 
            db_default = DBSCAN(eps = 0.003, min_samples = 3).fit(df_scaled)
            labels = db_default.labels_ 
            print(labels)

        
                    