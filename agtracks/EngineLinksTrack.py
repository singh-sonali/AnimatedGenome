from pygenometracks.tracks.GenomeTrack import GenomeTrack
from intervaltree import IntervalTree, Interval
import numpy as np


class EngineLinksTrack(GenomeTrack):
    SUPPORTED_ENDINGS = ['.arcs', '.arc' '.link', '.links']
    TRACK_TYPE = 'engine_links'
    OPTIONS_TXT = GenomeTrack.OPTIONS_TXT + """
# depending on the links type either and arc or a 'triangle' can be plotted. If an arc,
# a line will be drawn from the center of the first region, to the center of the other region.
# if a triangle, the vertix of the triangle will be drawn at the center between the two points (also the center of
# each position is used)
# links whose start or end is not in the region plotted are not shown.
# color of the lines
color = red
# for the links type, the options are arcs and triangles, the triangles option is convenient to overlay over a
# Hi-C matrix to highlight the matrix pixel of the highlighted link
links type = arcs
# if line width is not given, the score is used to set the line width
# using the following formula (0.5 * square root(score)
# line width = 0.5
# options for line style are 'solid', 'dashed', 'dotted' etc. The full list of
# styles can be found here: https://matplotlib.org/gallery/lines_bars_and_markers/linestyles.html
line style = solid
file_type = {}
    """.format(TRACK_TYPE)

    def __init__(self, *args, **kwarg):
        super(EngineLinksTrack, self).__init__(*args, **kwarg)
        if 'line width' not in self.properties:
            self.properties['line width'] = 0.5
        if 'line style' not in self.properties:
            self.properties['line style'] = 'solid'
        if 'links type' not in self.properties:
            self.properties['links type'] = 'arcs'
        
        self.background = True
        self.max_height = None
        
        if 'color' not in self.properties:
            self.properties['color'] = 'blue'

        if 'alpha' not in self.properties:
            self.properties['alpha'] = 0.8

    def plot(self, ax, chrom_region, region_start, region_end, links_list, count):
        """
        Makes and arc connecting two points on a linear scale representing
        interactions between Hi-C bins.
        :param ax: matplotlib axis
        :param label_ax: matplotlib axis for labels
        """

        if self.background == False:
            self.max_height = 0
            for x in range(count+1):
                start, end, score = links_list[x]
                if start == end:
                    start+=1
                if start> end:
                    start, end = end, start
                if 'line width' in self.properties:
                    self.line_width = float(self.properties['line width'])
                else:
                    self.line_width = 1.5
                # plots link only if interaction score is above some amount
                if score>= int(self.properties['links threshold']):
                    if self.properties['links type'] == 'triangles':
                        self.plot_triangles(ax, start, end)
                    else:
                        self.plot_arcs(ax, start, end)

            # the arc height is equal to the radius, the track height is the largest
            # radius plotted plus an small increase to avoid cropping of the arcs
            self.max_height += self.max_height * 0.1
            if 'orientation' in self.properties and self.properties['orientation'] == 'inverted':
                ax.set_ylim(self.max_height, -1)
            else:
                ax.set_ylim(-1, self.max_height)

            # self.log.debug('title is {}'.format(self.properties['title']))
        self.background = False

    def plot_y_axis(self, ax, plot_ax):
        pass

    def plot_arcs(self, ax, start, end):
        from matplotlib.patches import Arc

        diameter = (end - start)
        radius = float(diameter) / 2
        center = start + float(diameter) / 2
        if radius > self.max_height:
            self.max_height = radius
        ax.plot([center], [diameter])
        ax.add_patch(Arc((center, 0), diameter,
                         diameter, 0, 0, 180, color=self.properties['color'],
                         linewidth=self.line_width, ls=self.properties['line style']))

    def plot_triangles(self, ax, start, end):
        from matplotlib.patches import Polygon
        x1 = start
        x2 = x1 + float(end - start) / 2
        x3 = end
        y1 = 0
        y2 = (end - start)

        triangle = Polygon(np.array([[x1, y1], [x2, y2], [x3, y1]]), closed=False,
                           facecolor='none', edgecolor=self.properties['color'],
                           linewidth=self.line_width,
                           ls=self.properties['line style'])
        ax.add_artist(triangle)
        if y2 > self.max_height:
            self.max_height = y2