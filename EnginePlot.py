import sys
import numpy as np
import argparse
import logging
from configparser import ConfigParser
from ast import literal_eval
from pygenometracks._version import __version__
import time
from time import ctime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import matplotlib.textpath
import matplotlib.colors
import matplotlib.gridspec
import matplotlib.cm
import mpl_toolkits.axisartist as axisartist
import textwrap
from pygenometracks.utilities import file_to_intervaltree
from functools import partial

DEFAULT_BED_COLOR = '#1f78b4'
DEFAULT_BIGWIG_COLOR = '#33a02c'
DEFAULT_BEDGRAPH_COLOR = '#a6cee3'
DEFAULT_MATRIX_COLORMAP = 'Reds'
DEFAULT_TRACK_HEIGHT = 3  # in centimeters
# proportion of width dedicated to (figure, legends)
DEFAULT_FIGURE_WIDTH = 60  # in centimeters
# proportion of width dedicated to (figure, legends)
DEFAULT_WIDTH_RATIOS = (0.01, 0.90, 0.1)
DEFAULT_MARGINS = {'left': 0.04, 'right': 0.92, 'bottom': 0.03, 'top': 0.97}


from collections import OrderedDict
from agtracks import *
from pygenometracks.tracks import *
from pygenometracks.tracksClass import PlotTracks

FORMAT = "[%(levelname)s:%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s"
logging.basicConfig(format=FORMAT)
log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)


class EnginePlot(PlotTracks):
    def start(self, file_name, chrom, start, end, color_list, links_list, title=None):
        self.imgs = []
        # initializing variables needed for plotting
        self.track_height = self.get_tracks_height(start_region = start, end_region = end)
        self.title = title
        self.start = start
        self.end = end
        self.chrom = chrom
        self.color_list = color_list
        self.links_list = links_list
        self.background = True
        self.framecount = [-1]
        if self.fig_height:
            log.debug("Fig height is set")
        else:
            self.fig_height = sum(self.track_height)
        animate = self.func_animation()
    
    @staticmethod
    def get_available_tracks():
        avail_tracks = {}
        work = [GenomeTrack]
        while work:
            parent = work.pop()
            for child in parent.__subclasses__():
                if child not in avail_tracks:
                    track_type = child.TRACK_TYPE
                    avail_tracks[track_type] = child
                    work.append(child)
        return avail_tracks

    def get_tracks_height(self, start_region=None, end_region=None):
        """
        The main purpose of the following loop is
        to get the height of each of the tracks
        because for the Hi-C the height is variable with respect
        to the range being plotted, the function is called
        when each plot is going to be printed.

        Args:
            start_region: start of the region to plot. Only used in case the plot is a Hi-C matrix
            end_region: end of the region to plot. Only used in case the plot is a Hi-C matrix

        Returns:

        """
        track_height = []
        for track_dict in self.track_list:
            # if overlay previous is set to a value other than no
            # then, skip this track height
            if track_dict['overlay previous'] != 'no':
                continue
            elif 'x-axis' in track_dict and track_dict['x-axis'] is True:
                height = track_dict['fontsize'] / 10
            elif 'height' in track_dict:
                height = track_dict['height']
            # compute the height of a Hi-C track
            # based on the depth such that the
            # resulting plot appears proportional
            #
            #      /|\
            #     / | \
            #    /  |d \   d is the depth that we want to be proportional
            #   /   |   \  when plotted in the figure
            # ------------------
            #   region len
            #
            # d (in cm) =  depth (in bp) * width (in cm) / region len (in bp)

            elif 'depth' in track_dict and (track_dict['file_type'] == 'hic_matrix' or track_dict['file_type'] == 'engine_hic_matrix') :
                # to compute the actual width of the figure the margins and the region
                # set for the legends have to be considered
                # DEFAULT_MARGINS[1] - DEFAULT_MARGINS[0] is the proportion of plotting area

                hic_width = \
                    self.fig_width * (DEFAULT_MARGINS['right'] - DEFAULT_MARGINS['left']) * self.width_ratios[1]
                scale_factor = 0.6  # the scale factor is to obtain a 'pleasing' result.
                depth = min(track_dict['depth'], (end_region - start_region))

                height = scale_factor * depth * hic_width / (end_region - start_region)
            else:
                height = DEFAULT_TRACK_HEIGHT

            track_height.append(height)

        return track_height

    def plot(self, frame):
        """function for plotting each frame of animation 
           as updated tracks are drawn"""
        if frame%50 == 0:
            log.debug(ctime())
        self.framecount.append(frame)

        log.debug("Figure size in cm is {} x {}. Dpi is set to {}\n".format(self.fig_width, self.fig_height, self.dpi))
        
        # make sure that same frame is not being replotted
        if self.framecount[-1]!=self.framecount[-2]:
            log.info(frame)

            # empty out all fig axes
            if frame>0:
                plt.clf()

            if self.title:
                fig.suptitle(self.title)

            grids = matplotlib.gridspec.GridSpec(len(self.track_height), 3,
                                                 height_ratios=self.track_height,
                                                 width_ratios=self.width_ratios, wspace=0.01)
            axis_list = []
            # skipped_tracks is the count of tracks that have the
            # 'overlay previous' parameter and should be skipped
            skipped_tracks = 0
            plot_axis = None
            for idx, track in enumerate(self.track_obj_list):
                log.info("plotting {}".format(track.properties['section_name']))
                if idx == 0 and track.properties['overlay previous'] != 'no':
                    log.warn("First track can not have the `overlay previous` option")
                    track.properties['overlay previous'] = 'no'

                if track.properties['overlay previous'] in ['yes', 'share-y']:
                    overlay = True
                    skipped_tracks += 1
                else:
                    overlay = False

                if track.properties['overlay previous'] == 'share-y':
                    ylim = plot_axis.get_ylim()
                else:
                    idx -= skipped_tracks
                    plot_axis = axisartist.Subplot(fig, grids[idx, 1])
                    fig.add_subplot(plot_axis)
                    # turns off se lines around the tracks
                    plot_axis.axis[:].set_visible(False)
                    # to make the background transparent
                    plot_axis.patch.set_visible(False)

                    y_axis = plt.subplot(grids[idx, 0])
                    y_axis.set_axis_off()

                    label_axis = plt.subplot(grids[idx, 2])
                    label_axis.set_axis_off()

                plot_axis.set_xlim(self.start, self.end)

                # customized tracks require extra parameters
                if 'file_type' in track.properties and track.properties['file_type'] == 'engine_hic_matrix':
                        track.plot(plot_axis, self.chrom, self.start, self.end, color_list = self.color_list)
                elif 'file_type' in track.properties and track.properties['file_type'] == 'engine_links':
                        track.plot(plot_axis, self.chrom, self.start, self.end, self.links_list, (frame-1))
                else:
                    track.plot(plot_axis, self.chrom, self.start, self.end)
                track.plot_y_axis(y_axis, plot_axis)
                track.plot_label(label_axis)


                if track.properties['overlay previous'] == 'share-y':
                    plot_axis.set_ylim(ylim)

                if not overlay:
                    axis_list.append(plot_axis)

            if self.vlines_intval_tree:
                self.plot_vlines(axis_list, self.chrom, self.start, self.end)

            fig.subplots_adjust(wspace=0, hspace=0.0,
                                left=DEFAULT_MARGINS['left'],
                                right=DEFAULT_MARGINS['right'],
                                bottom=DEFAULT_MARGINS['bottom'],
                                top=DEFAULT_MARGINS['top'])

            # if not drawing first frame of animation, reduce size of updating list
            # if self.background == False:
            #     self.color_list = self.color_list[1:]
            self.background = False
            

        # returns updated figure to be used in function-based animation. if using artist animation, must return a list of iterables in the form of images (convert figure to image) to generate animation.
        fig.savefig('output.png', dpi=self.dpi)
        return fig,

    def artist_animation(self, imgs, interval = 300.0, dpi = 72, save_gif = True, saveto='testanimation.gif', show_gif = False):
        """builds animation from list of pre-plotted images. 
           best used when number of frames is small"""
        imgs = np.asarray(imgs)
        h, w, *c = imgs[0].shape
        fig, ax = plt.subplots(figsize=(np.round(w/dpi),np.round(h/dpi)))
        fig.subplots_adjust(bottom = 0,
                           top = 1,
                            right = 1,
                            left=0)
        ax.set_axis_off()
        axs = list(map(lambda x: [ax.imshow(x)], imgs))
        ani = animation.ArtistAnimation(fig, axs, interval=interval, repeat_delay=100, blit=True)

        if save_gif:
            ani.save(saveto, writer='imagemagick',dpi=dpi)

        return ani

    def func_animation(self):
        """builds animation by continously calling plot function. 
           optimal with large frame counts."""
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim = animation.FuncAnimation(fig=fig, func=self.plot, frames=2, interval = 1.0, blit = True)
        anim.save('diffhicanimation.mp4', writer=writer,dpi=72)
        
        


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description='Plots genomic tracks on specified region(s). '
                    'Citation : Ramirez et al.  High-resolution TADs reveal DNA '
                    'sequences underlying genome organization in flies. '
                    'Nature Communications (2018) doi:10.1038/s41467-017-02525-w',
        usage="%(prog)s --tracks tracks.ini --region chr1:1000000-4000000 -o image.png")

    parser.add_argument('--tracks',
                        help='File containing the instructions to plot the tracks. '
                        'The tracks.ini file can be genarated using the `make_tracks_file` program.',
                        type=argparse.FileType('r'),
                        required=True,
                        )
    parser.add_argument('--data', help='Text file containing Hi-C contacts (try using Juicebox straw to generate these).', type=argparse.FileType('r'),required=True)
    parser.add_argument('--loops', help='Bed file containing the locations of loops discovered.', type=argparse.FileType('r'),required=True)

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--region',
                       help='Region to plot, the format is chr:start-end')

    group.add_argument('--BED',
                       help='Instead of a region, a file containing the regions to plot, in BED format, '
                       'can be given. If this is the case, multiple files will be created using a prefix '
                       'the value of --outFileName',
                       type=argparse.FileType('r')
                       )

    parser.add_argument('--width',
                        help='figure width in centimeters',
                        type=float,
                        default=DEFAULT_FIGURE_WIDTH)

    parser.add_argument('--height',
                        help='Figure height in centimeters. If not given, the figure height is computed '
                             'based on the heights of the tracks. If given, the track height are proportionally '
                             'scaled to match the desired figure height.',
                        type=float)

    parser.add_argument('--title', '-t',
                        help='Plot title',
                        required=False)

    parser.add_argument('--outFileName', '-out',
                        help='File name to save the image, file prefix in case multiple images '
                             'are stored',
                        required=False)

    parser.add_argument('--vlines',
                        help='Genomic cooordindates separated by space. E.g. '
                        '--vlines 150000 3000000 124838433 ',
                        type=int,
                        nargs='+'
                        )

    parser.add_argument('--fontSize',
                        help='Font size for the labels of the plot',
                        type=float,
                        )

    parser.add_argument('--dpi',
                        help='Resolution for the image in case the'
                             'ouput is a raster graphics image (e.g png, jpg)',
                        type=int,
                        default=72
                        )

    parser.add_argument('--trackLabelFraction',
                        help='By default the space dedicated to the track labels is 0.05 of the'
                             'plot width. This fraction can be changed with this parameter if needed.',
                        default=0.05,
                        type=float)

    parser.add_argument('--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    return parser

def get_region(region_string):
    """
    splits a region string into
    a chrom, start_region, end_region tuple
    The region_string format is chr:start-end
    """
    if region_string:
        # separate the chromosome name and the location using the ':' character
        chrom, position = region_string.strip().split(":")

        # clean up the position
        for char in ",.;|!{}()":
            position = position.replace(char, '')

        position_list = position.split("-")
        try:
            region_start = int(position_list[0])
        except IndexError:
            region_start = 0
        try:
            region_end = int(position_list[1])
        except IndexError:
            region_end = 1e15 
        if region_start < 0:
            region_start = 0
        if region_end <= region_start:
            exit("Please check that the region end is larger than the region start.\n"
                 "Values given:\nstart: {}\nend: {}\n".format(region_start, region_end))

        return chrom, region_start, region_end


def main(args=None):
    args = parse_arguments().parse_args(args)
    import random
    color_list = []
    links_list = []
    file = open(args.data.name,'r')
    data = file.readlines()

    for line in data:
        start = (line.strip().split('\t')[0])
        end = (line.strip().split('\t')[1])
        score = (line.strip().split('\t')[2])
        color_list.append((int(start),int(end),float(score)))

    links = open(args.loops.name,'r')
    ldata = links.readlines()
    for line in ldata:
        start = (line.strip().split('\t')[1])
        end = (line.strip().split('\t')[2])
        score = (line.strip().split('\t')[4])
        links_list.append((int(start),int(end),float(score)))

    trp = EnginePlot(tracks_file=args.tracks.name, fig_width= args.width, fig_height=args.height, fontsize=args.fontSize, dpi=args.dpi, track_label_width=args.trackLabelFraction)

    if args.BED:
        count = 0
        for line in args.BED.readlines():
            count += 1
            try:
                chrom, start, end = line.strip().split('\t')[0:3]
            except ValueError:
                continue
            try:
                start, end = map(int, [start, end])
            except ValueError as detail:
                sys.stderr.write("Invalid value found at line\t{}\t. {}\n".format(line, detail))
            name = args.outFileName.split(".")
            file_suffix = name[-1]
            file_prefix = ".".join(name[:-1])

            file_name = "{}_{}-{}-{}.{}".format(file_prefix, chrom, start, end, file_suffix)
            if end - start < 200000:
                sys.stderr.write("A region shorter than 200kb has been "
                                 "detected! This can be too small to return "
                                 "a proper TAD plot!\n")
                # start -= 100000
                # start = max(0, start)
                # end += 100000
            sys.stderr.write("saving {}\n".format(file_name))
            print("{} {} {}".format(chrom, start, end))
            trp.start(file_name, chrom, start, end, color_list, title=args.title)

    else:
        region = get_region(args.region)
        trp.start(args.outFileName, *region, color_list, links_list, title=args.title)



if __name__ == "__main__":
    args = None
    # initialize figure used for plotting
    fig = plt.figure(figsize=(40.0,24.0))
    if len(sys.argv) == 1:
        args = ["--help"]
    main(args)











