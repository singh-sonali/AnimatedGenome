import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

from pygenometracks.tracks.GenomeTrack import GenomeTrack

class RegElementTrack(GenomeTrack):
	SUPPORTED_ENDINGS = ['.txt']
	TRACK_TYPE = 'regelement'
	OPTIONS_TXT = """
	height = 3
	title = 
	file = 
	file_type = regelement
	# when used with before a genes track, on the genes track:
	overlay_previous = yes 

	#File should be formatted like 
	#name	position	shape		
	#for each regulatory element.
	"""

	def __init__(self, *args, **kwargs):
		super(RegElementTrack, self).__init__(*args, **kwargs)
		line_number = 0
		self.elements_list = []
		with open(self.properties['file'], 'r') as file_h:
			for line in file_h.readlines():
				line_number +=1
				name, pos, shape = line.strip().split('\t')
				self.elements_list.append((name,pos,shape))

	def plot(self, ax, chrom_region, region_start, region_end):
		for element in self.elements_list:
			name, position, shape = element
			if shape == 'rectangle':
				startpos, endpos = position.strip().split('-')
				position = int(startpos)
			else:
				position = int(position)

			if region_start < position < region_end:
				if shape == 'oval':
					self.plot_ovals(ax, name, region_start, region_end, position)
				elif shape == 'hexagon':
					self.plot_hexagons(ax, name, region_start, region_end, position)
				else:
					self.plot_rectangles(ax, name, startpos, endpos)
			else:
				self.log.warn("File contains regulatory elements not contained in given region.")

	def plot_ovals(self, ax, name, region_start, region_end, pos):
		x = [region_start+500, region_end-500]
		y = [0.5, 0.5]
		ax.plot(x,y,'ko',alpha=0.3,color='grey',ls='solid')
		
		from matplotlib.patches import Ellipse
		width = (region_end-region_start)//100
		ylims = ax.get_ylim()
		height = (ylims[1] - ylims[0])/2
		ax.add_patch(Ellipse((pos,0.5),width,height,color='blue'))
		#ax.text(pos+width, (ylims[0]+ylims[1])/2, name, fontsize=self.properties['fontsize'])


	def plot_hexagons(self, ax, name, region_start, region_end, pos):
		from matplotlib.patches import Polygon
		xint = ((region_end-region_start)//90)//3
		xstart = pos - ((region_end-region_start)//180)
		xend = pos + ((region_end-region_start)//180)
		xlocs = [xstart, xstart+xint, xend-xint, xend, xend-xint, xstart+xint]

		ylims = ax.get_ylim()
		yint = (ylims[1] - ylims[0])/4
		ylow, ymid, yhigh = ylims[0], ylims[0]+yint, (ylims[0]+ylims[1])/2
		ylocs = [ymid, yhigh, yhigh, ymid, ylow, ylow]

		vertices = []
		for i in range(len(ylocs)):
			x = xlocs[i]
			y = ylocs[i]
			vertices.append((x,y))
		vertices = np.array(vertices)
		ax.add_patch(Polygon(vertices,closed=True, fill=True, color='red'))

	def plot_rectangles(self, ax, name, startpos, endpos):
		from matplotlib.patches import Rectangle 
		ylims = ax.get_ylim()
		y = ylims[0]
		h = ylims[1] - ylims[0]
		x = int(startpos)
		w = int(endpos)-x
		ax.add_patch(Rectangle((x,y),w,h,color='gray',alpha=0.4))

	def plot_y_axis(self, ax, plot_ax):
		pass









