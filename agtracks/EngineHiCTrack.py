from hicmatrix import HiCMatrix
import hicmatrix.utilities
import scipy.sparse
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, dia_matrix, triu, tril, rand

from pygenometracks.tracks.GenomeTrack import GenomeTrack

DEFAULT_MATRIX_COLORMAP = 'RdYlBu_r'
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

class EngineHiCTrack(GenomeTrack):
    SUPPORTED_ENDINGS = ['.h5', '.cool' '.mcool']
    TRACK_TYPE = 'engine_hic_matrix'
    OPTIONS_TXT = """
title =
# The different options for color maps can be found here: https://matplotlib.org/users/colormaps.html
# the default color map is RdYlBu_r (_r) stands for reverse
#colormap = RdYlBu_r
# depth is the maximum distance that should be plotted.
depth = 100000
# height of track (in cm) can be given. Otherwise, the height is computed such that the proportions of the
# hic matrix are kept (e.g. the image does not appear shrink or extended)
# height = 10
# min_value and max_value refer to the contacts in the matrix.
#min_value =2.8
#max_value = 3.0
# the matrix can be transformed using the log1 (or log, but zeros could be problematic)
transform = log1p
# show masked bins plots as white lines
# those bins that were not used during the correction
# the default is to extend neighboring bins to
# obtain an aesthetically pleasant output
show_masked_bins = no
# if you want to plot the track upside-down:
# orientation = inverted
# optional if the values in the matrix need to be scaled the
# following parameter can be used. This is useful to plot multiple hic-matrices on the same scale
# scale factor = 1
file_type = {}
    """.format(TRACK_TYPE)

    def __init__(self, *args, **kwargs):
        super(EngineHiCTrack, self).__init__(*args, **kwargs)

        log.debug('FILE {}'.format(self.properties))
        # log.debug('pRegion {}'.format(pRegion))
        region = None
        if self.properties['region'] is not None:
            if self.properties['region'][2] == 1e15:
                region = [str(self.properties['region'][0])]
            elif len(self.properties['region']) == 3:
                start = int(self.properties['region'][1]) - int(self.properties['depth'])
                if start < 0:
                    start = 0
                end = int(self.properties['region'][2]) + int(self.properties['depth'])

                region = [str(self.properties['region'][0]) + ':' + str(start) + '-' + str(end)]
       
        # initialize matrix as HiCMatrix object with no data
        self.hic_ma = HiCMatrix.hiCMatrix(pMatrixFile=None, pChrnameList=region)
        # create matrix to fill out data and intervals 
        if 'matrix shape' not in self.properties:
            self.properties['matrix shape'] = 1000
        if 'binsize' not in self.properties:
            self.properties['binsize'] = 3000
        if 'intervals start' not in self.properties:
            self.properties['intervals start'] = 0
            
        self.hic_ma.matrix, self.hic_ma.cut_intervals = \
            self.definematrix(self.properties['matrix shape'], self.properties['binsize'], self.properties['intervals start'], self.properties['chrom'])

        self.hic_ma.interval_trees, self.hic_ma.chrBinBoundaries = \
            self.hic_ma.intervalListToIntervalTree(self.hic_ma.cut_intervals)

        if len(self.hic_ma.matrix.data) == 0:
            self.log.error("Matrix {} is empty".format(self.properties['file']))
            exit(1)
        if 'show_masked_bins' in self.properties and self.properties['show_masked_bins'] == 'yes':
            pass
        else:
            self.hic_ma.maskBins(self.hic_ma.nan_bins)

        # check that the matrix can be log transformed
        if 'transform' in self.properties:
            if self.properties['transform'] == 'log1p':
                if self.hic_ma.matrix.data.min() + 1 < 0:
                    self.log.error("\n*ERROR*\nMatrix contains negative values.\n"
                                   "log1p transformation can not be applied to \n"
                                   "values in matrix: {}".format(self.properties['file']))
                    exit(1)

            elif self.properties['transform'] == '-log':
                if self.hic_ma.matrix.data.min() < 0:
                    self.log.error("\n*ERROR*\nMatrix contains negative values.\n"
                                   "log(-1 * <values>) transformation can not be applied to \n"
                                   "values in matrix: {}".format(self.properties['file']))
                    exit(1)

            elif self.properties['transform'] == 'log':
                if self.hic_ma.matrix.data.min() < 0:
                    self.log.error("\n*ERROR*\nMatrix contains negative values.\n"
                                   "log transformation can not be applied to \n"
                                   "values in matrix: {}".format(self.properties['file']))
                    exit(1)

    
        binsize = self.hic_ma.getBinSize()
        max_depth_in_bins = int(self.properties['depth'] / binsize)

        # work only with the lower matrix
        # and remove all pixels that are beyond
        # 2 * max_depth_in_bis which are not required
        # (this is done by subtracting a second sparse matrix
        # that contains only the lower matrix that wants to be removed.
        limit = 2 * max_depth_in_bins
        self.hic_ma.matrix = scipy.sparse.triu(self.hic_ma.matrix, k=0, format='csr') - \
            scipy.sparse.triu(self.hic_ma.matrix, k=limit, format='csr')
        self.hic_ma.matrix.eliminate_zeros()

        # fill the main diagonal, otherwise it looks
        # not so good. The main diagonal is filled
        # with an array containing the max value found
        # in the matrix
        if sum(self.hic_ma.matrix.diagonal()) == 0:
            self.log.info("Filling main diagonal with max value because it empty and looks bad...\n")
            max_value = self.hic_ma.matrix.data.max()
            main_diagonal = scipy.sparse.dia_matrix(([max_value] * self.hic_ma.matrix.shape[0], [0]),
                                                    shape=self.hic_ma.matrix.shape)
            self.hic_ma.matrix = self.hic_ma.matrix + main_diagonal

        self.plot_inverted = False
        if 'orientation' in self.properties and self.properties['orientation'] == 'inverted':
            self.plot_inverted = True

        self.norm = None

        if 'colormap' not in self.properties:
            self.properties['colormap'] = DEFAULT_MATRIX_COLORMAP
        self.cmap = cm.get_cmap(self.properties['colormap'])
        self.cmap.set_bad('white')
        #self.cmap.set_over('blue')
        self.background = True


    def plot(self, ax, chrom_region, region_start, region_end, color_list):
        log.debug('chrom_region {}, region_start {}, region_end {}'.format(chrom_region, region_start, region_end))
        chrom_sizes = self.hic_ma.get_chromosome_sizes()
        if chrom_region not in chrom_sizes:
            chrom_region_before = chrom_region
            chrom_region = self.change_chrom_names(chrom_region)
            if chrom_region not in chrom_sizes:
                self.log.error("*Error*\nNeither " + chrom_region_before + " "
                               "nor " + chrom_region + " exits as a chromosome"
                               " name on the matrix.\n")
                return

        chrom_region = self.check_chrom_str_bytes(chrom_sizes, chrom_region)
        if region_end > chrom_sizes[chrom_region]:
            self.log.error("*Error*\nThe region to plot extends beyond the chromosome size. Please check.\n")
            self.log.error("{} size: {}. Region to plot {}-{}\n".format(chrom_region, chrom_sizes[chrom_region],
                                                                        region_start, region_end))
        # background is a fixed domain spanning region length 
        if self.background == True:
            domain = self.isolate_domain(chrom_region, region_start, region_end)

        # color cell of observed interaction to create updated matrix data for plotting
        first_bin,second_bin = self.color_cell(chrom_region, color_list)
        # get bin id of start and end of region in given chromosome
        chr_start_id, chr_end_id = self.hic_ma.getChrBinRange(chrom_region)
        chr_start = self.hic_ma.cut_intervals[chr_start_id][1]
        chr_end = self.hic_ma.cut_intervals[chr_end_id - 1][1]
        start_bp = max(chr_start, region_start - self.properties['depth'])
        end_bp = min(chr_end, region_end + self.properties['depth'])

        idx, start_pos = list(zip(*[(idx, x[1]) for idx, x in
                                    enumerate(self.hic_ma.cut_intervals)
                                    if x[0] == chrom_region and x[1] >= start_bp and x[2] <= end_bp]))

        idx = idx[0:-1]
        # select only relevant matrix part
        matrix = self.hic_ma.matrix[idx, :][:, idx]
        # limit the 'depth' based on the length of the region being viewed

        region_len = region_end - region_start
        depth = min(self.properties['depth'], int(region_len * 1.25))
        depth_in_bins = int(1.5 * region_len / self.hic_ma.getBinSize())

        if depth < self.properties['depth']:
            # remove from matrix all data points that are not visible.
            matrix = matrix - scipy.sparse.triu(matrix, k=depth_in_bins, format='csr')
        matrix = np.asarray(matrix.todense().astype(float))
        if 'scale factor' in self.properties:
            matrix = matrix * self.properties['scale factor']

        if 'transform' in self.properties:
            if self.properties['transform'] == 'log1p':
                matrix += 1
                self.norm = colors.LogNorm()

            elif self.properties['transform'] == '-log':
                mask = matrix == 0
                try:
                    matrix[mask] = matrix[mask == False].min()
                    matrix = -1 * np.log(matrix)
                except ValueError:
                    self.log.info('All values are 0, no log applied.')

            elif self.properties['transform'] == 'log':
                mask = matrix == 0
                try:
                    matrix[mask] = matrix[mask == False].min()
                    matrix = np.log(matrix)
                except ValueError:
                    self.log.info('All values are 0, no log applied.')

            elif self.properties['transform'] == 'symlog':
                self.norm = colors.SymLogNorm(linthresh=0.03,linscale=0.03)

        if 'max_value' in self.properties and self.properties['max_value'] != 'auto':
            vmax = self.properties['max_value']

        else:
            # try to use a 'aesthetically pleasant' max value

            vmax = np.percentile(matrix.diagonal(1), 80)

        if 'min_value' in self.properties and self.properties['min_value'] != 'auto':
            vmin = self.properties['min_value']
        else:
            if depth_in_bins > matrix.shape[0]:
                depth_in_bins = matrix.shape[0] - 5

            # if the region length is large with respect to the chromosome length, the diagonal may have
            # very few values or none. Thus, the following lines reduce the number of bins until the
            # diagonal is at least length 5
            num_bins_from_diagonal = int(region_len / self.hic_ma.getBinSize())
            for num_bins in range(0, num_bins_from_diagonal)[::-1]:
                distant_diagonal_values = matrix.diagonal(num_bins)
                if len(distant_diagonal_values) > 5:
                    break

            vmin = np.median(distant_diagonal_values)

        # self.log.info("setting min, max values for track {} to: {}, {}\n".
        #               format(self.properties['section_name'], vmin, vmax))
        self.img = self.pcolormesh_45deg(ax, matrix, start_pos, first_bin, second_bin, vmax=vmax, vmin=vmin)
        self.img.set_rasterized(True)
        if self.plot_inverted:
            ax.set_ylim(depth, 0)
        else:
            ax.set_ylim(0, depth)


    def plot_y_axis(self, cbar_ax, plot_ax):
        if 'transform' in self.properties and \
                self.properties['transform'] in ['log', 'log1p']:
            # get a useful log scale
            # that looks like [1, 2, 5, 10, 20, 50, 100, ... etc]

            # The following code is problematic with some versions of matplotlib.
            # Should be uncommented once the problem is clarified
            from matplotlib.ticker import LogFormatter
            formatter = LogFormatter(10, labelOnlyBase=False)
            aa = np.array([1, 2, 5])
            tick_values = np.concatenate([aa * 10 ** x for x in range(10)])
            try:
                cobar = plt.colorbar(self.img, ticks=tick_values, format=formatter, ax=cbar_ax, fraction=0.95)
            except AttributeError:
                return
        else:
            try:
                cobar = plt.colorbar(self.img, ax=cbar_ax, fraction=0.95)
            except AttributeError:
                return

        cobar.solids.set_edgecolor("face")
        cobar.ax.tick_params(labelsize='smaller')
        cobar.ax.yaxis.set_ticks_position('left')

        # adjust the labels of the colorbar
        labels = cobar.ax.get_yticklabels()
        ticks = cobar.ax.get_yticks()

        if ticks[0] == 0:
            # if the label is at the start of the colobar
            # move it above avoid being cut or overlapping with other track
            labels[0].set_verticalalignment('bottom')
        if ticks[-1] == 1:
            # if the label is at the end of the colobar
            # move it a bit inside to avoid overlapping
            # with other labels
            labels[-1].set_verticalalignment('top')

        cobar.ax.set_yticklabels(tick_values)

        

    def pcolormesh_45deg(self, ax, matrix_c, start_pos_vector, firstbin, secondbin, vmin=None, vmax=None):
        """
        Turns the matrix 45 degrees and adjusts the
        bins to match the actual start end positions.
        """
        import itertools
        # code for rotating the image 45 degrees
        n = matrix_c.shape[0]
        # create rotation/scaling matrix
        t = np.array([[1, 0.5], [-1, 0.5]])
        # create coordinate matrix and transform it
        matrix_a = np.dot(np.array([(i[1], i[0])
                                    for i in itertools.product(start_pos_vector[::-1],
                                                               start_pos_vector)]), t)
        # this is to convert the indices into bp ranges
        x = matrix_a[:, 1].reshape(n + 1, n + 1)
        y = matrix_a[:, 0].reshape(n + 1, n + 1)
        
        # If interested in only coloring one cell at a time and annotating said cell, the following should be uncommented.
        # if self.background == False and firstbin != -1:
            # row, col = (n-firstbin)-1, secondbin
            # annotate_x = x[row+1, col]
            # annotate_y = y[row+1, col]
            # val = matrix_c[firstbin, secondbin]
            # text = self.annotate_heatmap(ax, annotate_x, annotate_y, val)

            #plot only colored cell
            # x = x[row:(row+2), col: (col+2)]
            # y = y[row:(row+2), col: (col+2)]
            # matrix_c = np.array([[0],[val]])

        # plot
        im = ax.pcolormesh(x, y, np.flipud(matrix_c),
                           vmin=vmin, vmax=vmax, cmap=self.cmap, edgecolors='face', norm=self.norm)

        self.background = False

        return im

    def isolate_domain(self, chrname, start, end):
        """
        Colors triangular portion of matrix that has cells corresponding with isolated domain (dependent on region start and end)
        """
        start_bin, end_bin = self.hic_ma.getRegionBinRange(chrname, start, end)
        start_bin+=1
        end_bin-=1
        row_start = []
        data_shape = self.hic_ma.matrix.data.shape
        self.hic_ma.matrix.data = np.zeros(data_shape)

        for start_pos in range(start_bin, end_bin + 2):
            row_start.append(self.hic_ma.matrix.indptr[start_pos])
        
        for row in range(start_bin, end_bin+1):
            for cell in range(row, end_bin+1):
                indices_range = self.hic_ma.matrix.indices[row_start[0]:row_start[1]]
                col = [i for i, value in enumerate(indices_range) if value == cell]
                if col:
                    cell_loc = row_start[0] + col[0]
                    # sets domain cell colors to base
                    self.hic_ma.matrix.data[cell_loc] = 0.5
                    col = []
            row_start = row_start[1:]
            if len(row_start) == 1:
                break

        # data values not in domain are set as "bad values" and will be colored white in plotting
        self.hic_ma.matrix.data[self.hic_ma.matrix.data==0]=np.nan

    def color_cell(self, chrname, color_list):
        """
        Colors cells in color_list that indicate interaction.
        """
        if self.background == True or not color_list:
            first_bin,second_bin = -1, -1
        elif color_list and self.background == False:
            while color_list:
                coordinates = color_list[0]
                loc1 = coordinates[0]
                loc2 = coordinates[1]
                if loc1>loc2:
                    loc1, loc2 = loc2, loc1
                first_bin, second_bin = self.hic_ma.getRegionBinRange(chrname,loc1,loc2)
                col = second_bin
                row_start = self.hic_ma.matrix.indptr[first_bin]
                row_end = self.hic_ma.matrix.indptr[first_bin + 1]

                indices_range = self.hic_ma.matrix.indices[row_start:row_end]
                res_list = [i for i, value in enumerate(indices_range) if value==col]

                if res_list:
                    cell_loc = row_start + res_list[0]
                    if len(coordinates) == 3:
                        # color is set based on interaction score
                        self.hic_ma.matrix.data[cell_loc] += coordinates[2]
                        # if len(color_list) == 1:
                        #     self.hic_ma.matrix.data[cell_loc]= 10001
                    else: 
                        self.hic_ma.matrix.data[cell_loc] = 1000
                color_list = color_list[1:]
        return first_bin, second_bin


    def annotate_heatmap(self, ax, x, y, val):
        """
        Displays matrix value of cell on heatmap
        """
        binsize = self.hic_ma.getBinSize()
        text = ax.text(x+(binsize/2), y+(binsize/4), val, ha="center", va="center", color="white", fontsize = "12")
        return text

    def definematrix(self, shape, binsize, intervals_start,chrom):
        """
        Creates modifiable Hi-C matrix based on given shape and binsize. End position of region plotted must be less than 
        shape*binsize.
        """
        shape = int(shape)
        sparse_matrix = rand(shape,shape, density = 1, dtype=float)
        matrix = csr_matrix(sparse_matrix)
        cut_intervals = []
        binsize = int(binsize)
        intervals_start = int(intervals_start)
        for x in range(sparse_matrix.shape[0]):
            interval = (chrom, binsize*x+intervals_start, binsize*(x+1)+intervals_start, 1.0)
            cut_intervals.append(interval)
       
        return matrix, cut_intervals



