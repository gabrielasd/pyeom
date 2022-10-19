from matplotlib import pyplot
import matplotlib as mpl


pyplot.rc('font', family='serif')
pyplot.rc('xtick', labelsize='x-small')
pyplot.rc('ytick', labelsize='x-small')

scvname = '$results'
NPLOTS = $nplots
XVALS = $xvals
xlabel = '$xlabel'
ylabel = '$ylabel'
title = '$title'
xlims = (xmin, xmax)
ylims = (ymin, ymax)


markers = ['o', '>', 's', '*', '<']
# colors = ['r', 'g', 'b', 'k', 'm']

results, labels = get_results(scvname)
assert len(results) == NPLOTS
assert len(markers) >= NPLOTS

# Create figure and add axis
fig = pyplot.figure(figsize=(8, 6))
ax = fig.add_subplot(1, NPLOTS)

for i in range(nplots):
    ax[i].plot(XVALS, results[i], marker=markers[i], label=labels[i])

# Set the x axis
ax.set_xlabel(xlabel)
# Set the y axis label of the current axis.
ax.set_ylabel(ylabel)
# Set a title of the current axes.
ax.set_title(title)
# show a legend on the plot
ax.legend(bbox_to_anchor=(1.04, 1), frameon=False, fontsize=10, borderaxespad=0) #, loc=1
# Edit the major and minor ticks of the x and y axes
ax.xaxis.set_tick_params(which='major', size=7, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=4, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=7, width=1, direction='out', right='on')
ax.yaxis.set_tick_params(which='minor', size=4, width=1, direction='out', right='on')
# Set the axis limits
if lims is not None:
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
# Edit the major and minor tick locations
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
# Display a figure.
pyplot.show()