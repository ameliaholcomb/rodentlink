import struct
import sys
import math
import frames

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

	for frame in framesList:
		print(frame)
