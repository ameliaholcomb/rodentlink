import math
import struct
from collections import namedtuple

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

class ParseError(Exception):
	def __init__(self, errorString):
		self.value = errorString

class TLVType(object):
	"""TLVType classes hold the content for a TLV.
		They should be initialized with a buffer containing exactly
		as much data as the TLV header indicates is contained in the content.
		As a result, they do not pass any data back to the caller and have
		only an __init__ and __str__ method.
	"""
	pass

class DetectedObjects(TLVType):
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
				raise ParseError('Could not parse object with tuple {}: {}'.format(objTuple, e))

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
			raise ParseError("Received more data than expected. Indicates earlier parsing error.")

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


class RangeProfile(TLVType):
	RANGE_LEN = 256

	def __init__(self, data):
		self.range = {}
		for i in range(self.RANGE_LEN):
			rangeProf = struct.unpack('H', data[2*i:2*i+2])
			self.range[i] = rangeProf[0] * 1.0 * 6 / 8  / (1 << 8)
	def __str__(self):
		result = "\tRange Profile %d\n"%(RANGE_PROFILE)
		for k, v in self.range.items():
			result += "\t\t[%d]:\t%07.3f\n"%(k, v)
		return result


class Stats(TLVType):
	def __init__(self, data):
		vals = struct.unpack('6I', data[:STATS_BYTES])
		self.interProcess = vals[0]
		self.transmitOut = vals[1]
		self.frameMargin = vals[2]
		self.chirpMargin = vals[3]
		self.activeCPULoad = vals[4]
		self.interCPULoad = vals[5]

	def __str__(self):
		result = "\tOutputMsgStats:\t%d\n"%(STATS)
		result += "\t\tInterprocess:\t%d\n"%(self.interProcess)
		result += "\t\tTransmitOut:\t%d\n"%(self.transmitOut)
		result += "\t\tFrameMargin:\t%d\n"%(self.frameMargin)
		result += "\t\tChirpMargin:\t%d\n"%(self.chirpMargin)
		result += "\t\tActiveCPULoad:\t%d\n"%(self.activeCPULoad)
		result += "\t\tInterCPULoad:\t%d\n"%(self.interCPULoad)
		return result

class TLV(object):
	"""Class to parse and save a TLV header and content.
		Each Parse method returns the remaining buffer, without the parsed content.
		Usage:
			tlv = TLV()
			data = tlv.ParseHeader(data)
			data = tlv.ParseContents(data)
	"""

	def __init__(self):
		self.header = None 		# TLVHeader namedtuple
		self.contents = None	# TLVType

	def ParseHeader(self, data):
		"""Parses TLV header from start of data.
			Returns remaining (post-header) data.
		"""
		try:
			self.header = TLVHeader(*struct.unpack('2I', data[:TLV_HEADER_BYTES]))
			return data[TLV_HEADER_BYTES:]
		except:
			raise ParseError('Improper TLV header: {}'.format(data[:TLV_HEADER_BYTES]))

	def ParseContents(self, data):
		"""Parses TLV objects from start of data, according to self.header.type.

			Fails if ParseHeader() has not yet been called.
			Returns remaining data.
		"""
		if not self.header:
			raise ParseError('Must call ParseHeader before ParseObjects')
		if self.header.type == DETECTED_OBJECTS:
			self.contents = DetectedObjects(data[:self.header.length])
		elif self.header.type == RANGE_PROFILE:
			self.contents = RangeProfile(data[:self.header.length])
		elif self.header.type == STATS:
			self.contents = Stats(data[:self.header.length])
		else:
			raise ParseError('Unknown TLV Type: {}'.format(self.header.type))

		return data[self.header.length:]

	def __str__(self):
		return self.contents.__str__()


class Frame(object):
	"""Class to parse and save a Frame and its TLVs.
		Each Parse method returns the remaining buffer, without the parsed content.
		Usage:
			frame = Frame()
			data = frame.ParseHeader(data)
			data = frame.ParseTLVs(data)
	"""

	def __init__(self):
		self.header = None		# FrameHeader namedtuple
		self.tlvs = []			# List of TLV objects

	def ParseHeader(self, data):
		"""Parses frame header from start of data.
			Returns remaining (post-header) data.
		"""
		try:
			self.header = FrameHeader(*struct.unpack('Q7I', data[:FRAME_HEADER_BYTES]))
			return data[FRAME_HEADER_BYTES:]

		except:
			raise ParseError('Improper Frame header: {}'.format(data[:FRAME_HEADER_BYTES]))

	def ParseTLVs(self, data):
		"""Parses self.header.numTLV TLV's from start of data.

			Fails if ParseHeader() has not yet been called.
			Returns remaining data.
		"""
		if not self.header:
			raise ParseError('Must call ParseHeader before ParseTLVs')

		for i in range(self.header.numTLVs):
			tlv = TLV()
			data = tlv.ParseHeader(data)
			print(tlv.header.type)
			data = tlv.ParseContents(data)
			self.tlvs.append(tlv)

		return data

	def __str__(self):
		result = "Packet ID:\t%d\n"%(self.header.frameNum)
		result += "Length:\t\t%d\n"%(self.header.length)
		result += "Version:\t%x\n"%(self.header.version)
		result += "TLV:\t\t%d\n"%(self.header.numTLVs)
		result += "Detect Obj:\t%d\n"%(self.header.numObj)
		result += "Platform:\t%X\n"%(self.header.platform)
		for tlv in self.tlvs:
			result += tlv.__str__()
		return result
