import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# from utils.paths import paths
from ..utils.functions import fig
# from utils.setup import *
from ..utils.load import *
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pandas as pd
#DAC code

def rose_plot(ax, angles, bins=30, density=None, offset=0, lab_unit="degrees",
              start_zero=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    # Wrap angles to [-pi, pi)
    angles = (angles+180)*np.pi/180
    angles = (angles + np.pi) % (2*np.pi) - np.pi
    
    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
           edgecolor='black', fill=True, linewidth=.4)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)
        
def plot_eye_movements(all_fixations,all_saccades):
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # Plot main sequence
    axs[0, 0].scatter(all_saccades['sac_vmax'], all_saccades['sac_amplitude'],s=1)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xlabel('Saccade Amplitude')
    axs[0, 0].set_ylabel('Saccade Peak Velocity')
    axs[0, 0].set_title('Main Sequence',fontweight='bold')

    # Plot  saccades amplitude
    axs[0, 1].hist(all_saccades['sac_amplitude'], bins=50, edgecolor='black', fill=True, linewidth=.4)
    axs[0, 1].set_xlabel('saccades Amplitude')
    axs[0, 1].set_ylabel('Cases')
    axs[0, 1].set_title('Saccades:  Amplitude',fontweight='bold')

    # Plot fixation distribution
    n, bins, _ = axs[1, 0].hist(all_fixations['duration'],  bins=100, edgecolor='black', fill=True, linewidth=.4)
    #axs[0, 2].hist(saccades_tmp['duration'], bins=100, alpha=0.5, label='Fixation Durations')
    axs[1, 0].set_xlabel('Duration [ms]')
    axs[1, 0].set_ylabel('Cases')
    axs[1, 0].set_xlim([0,800])
    axs[1, 0].set_title('Fixations: Duration',fontweight='bold')

    axs[0, 2].remove()
    # Plot angular histograms of saccades
    axs[0, 2] = plt.subplot( 2,3,3,projection='polar')
    rose_plot(axs[0,2],  all_saccades['sac_angle'])# ,density=False)
    axs[0, 2].set_xlabel('Saccade Angle (degrees)')
    axs[0, 2].set_title('Angular Histogram of Saccades',fontweight='bold')

    # Plot fixations locations
    axs[1, 2].scatter(all_fixations['fix_avgpos_x'], all_fixations['fix_avgpos_y'],s=1)
    axs[1, 2].set_xlabel('Horizontal Position [pix]')
    axs[1, 2].set_ylabel('Vertical Position [pix]')
    axs[1, 2].set_title('Fixations: Location',fontweight='bold')
    axs[1, 2].set_aspect('equal')

    # Plot heatmap
    fixations_x = all_fixations['fix_avgpos_x']
    fixations_y = all_fixations['fix_avgpos_y']
    # Calculate the 2D histogram of fixations
    heatmap, xedges, yedges = np.histogram2d(fixations_y, fixations_x, bins=120)
    axs[1, 1].imshow(heatmap.T, cmap='hot', origin='lower', extent=[0, 1280, 1024, 0])#extent=[ yedges[0], yedges[-1],xedges[0], xedges[-1]])
    axs[1, 1].set_xlabel('Horizontal Position [pix]')
    axs[1, 1].set_ylabel('Vertical Position [pix]')
    axs[1, 1].set_title('Fixations: Heatmap')
    axs[1, 1].set_title('Heatmap',fontweight='bold')# Remove empty subplot

    #fig.delaxes(axs[1, 1])
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    #n, bins, _ = plt.hist(all_fixations['duration'], bins=100, edgecolor='black', fill=True, linewidth=0.4)

    # Find the index of the bin with the maximum frequency
    peak_index = np.argmax(n)

    # Get the corresponding time duration from the bin edges
    peak_duration = (bins[peak_index] + bins[peak_index + 1]) / 2

    # Print the time duration at which the distribution peaks
    print(f"The distribution peaks at a duration of {peak_duration:.2f} ms.")
    mean_duration = all_fixations['duration'].mean()
    std_duration = np.std(all_fixations['duration'])
    print(f"Mean Duration: {mean_duration:.2f} ms ± {std_duration:.2f} ms")   
    
 # Calculate the average number of fixations per trial per subject
    average_fixations_per_trial_per_subject = all_fixations.groupby(['sub_id', 'trial'])['duration'].count().groupby('sub_id').mean()
    average_fixations_per_trial = all_fixations.groupby(['sub_id', 'trial'])['duration'].count().mean()
    std_fixtr = np.std(all_fixations.groupby(['sub_id', 'trial'])['duration'].count())
    
    # Print the result
    print(f"Average number of fixations per trial per subject: {average_fixations_per_trial:.2f} ± {std_fixtr:.2f}")
    print(average_fixations_per_trial_per_subject)
    return fig

def plot_eye_movements_paper(all_fixations,all_saccades,all_fixations2,all_saccades2):
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a figure and subplots
    fig, axs = plt.subplots(2, 5, figsize=(15, 8))

    # Plot main sequence 1
    # axs[0, 0].scatter(all_saccades['sac_vmax'], all_saccades['sac_amplitude'],s=1)
    # axs[0, 0].set_xscale('log')
    # axs[0, 0].set_yscale('log')
    # axs[0, 0].set_xlabel('Saccade Amplitude')
    # axs[0, 0].set_ylabel('Saccade Peak Velocity')
    # axs[0, 0].set_title('Main Sequence',fontweight='bold')
#############################
    msheatmap, msxedges, msyedges = np.histogram2d(all_saccades['sac_amplitude'], all_saccades['sac_vmax'], bins=600)
    axs[0, 0].imshow(msheatmap.T, cmap='hot', origin='lower', extent=[msxedges[0], msxedges[-1], msyedges[0], msyedges[-1]])

    axs[0, 0].set_ylim([15,8e2])
    axs[0, 0].set_xlim([5e-2,10])
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xlabel('Saccade Amplitude')
    axs[0, 0].set_ylabel('Saccade Peak Velocity')
    axs[0, 0].set_title('Main Sequence',fontweight='bold')
#############################
    # Plot main sequence 2
    msheatmap2, msxedges2, msyedges2 = np.histogram2d(all_saccades2['sac_amplitude'], all_saccades2['sac_vmax'], bins=600)
    axs[1, 0].imshow(msheatmap2.T, cmap='hot', origin='lower', extent=[msxedges2[0], msxedges2[-1], msyedges2[0], msyedges2[-1]])

    axs[1, 0].set_ylim([15,8e2])
    axs[1, 0].set_xlim([5e-2,10])
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xlabel('Saccade Amplitude')
    axs[1, 0].set_ylabel('Saccade Peak Velocity')
    axs[1, 0].set_title('Main Sequence',fontweight='bold')

    # Plot  saccades amplitude
    axs[0, 1].hist(all_saccades['sac_amplitude'], bins=50, edgecolor='black', fill=True, linewidth=.4,density=True)
    axs[0, 1].set_xlabel('saccades Amplitude')
    axs[0, 1].set_ylabel('Probability')
    axs[0, 1].set_title('Saccades:  Amplitude',fontweight='bold')


    # Plot  saccades amplitude2
    axs[1, 1].hist(all_saccades2['sac_amplitude'], bins=50, edgecolor='black', fill=True, linewidth=.4,density=True)
    axs[1, 1].set_xlabel('saccades Amplitude')
    axs[1, 1].set_ylabel('Probability')
    axs[1, 1].set_title('Saccades:  Amplitude',fontweight='bold')
    
    # Plot fixation distribution
    n1, bins1, _ = axs[0, 2].hist(all_fixations['duration'],  bins=100, edgecolor='black', fill=True, linewidth=.4,density=True)
    #axs[0, 2].hist(saccades_tmp['duration'], bins=100, alpha=0.5, label='Fixation Durations')
    axs[0, 2].set_xlabel('Duration [ms]')
    axs[0, 2].set_ylabel('Probability')
    axs[0, 2].set_xlim([0,800])
    axs[0, 2].set_title('Fixations: Duration',fontweight='bold')


    # Plot fixation distribution 2
    n2, bins2, _ = axs[1, 2].hist(all_fixations2['duration'],  bins=100, edgecolor='black', fill=True, linewidth=.4,density=True)
    #axs[0, 2].hist(saccades_tmp['duration'], bins=100, alpha=0.5, label='Fixation Durations')
    axs[1, 2].set_xlabel('Duration [ms]')
    axs[1, 2].set_ylabel('Probability')
    axs[1, 2].set_xlim([0,800])
    axs[1, 2].set_title('Fixations: Duration',fontweight='bold')

    # Plot angular histogram

    axs[0, 4].remove()
    # Plot angular histograms of saccades
    axs[0, 4] = plt.subplot( 2,5,5,projection='polar')
    rose_plot(axs[0,4],  all_saccades['sac_angle'])# ,density=False)
    axs[0, 4].set_xlabel('Saccade Angle (degrees)')
    axs[0, 4].set_title('Angular Histogram of Saccades',fontweight='bold')

    # Plot angular histogram 2

    axs[1, 4].remove()
    # Plot angular histograms of saccades
    axs[1, 4] = plt.subplot( 2,5,10,projection='polar')
    rose_plot(axs[1,4],  all_saccades2['sac_angle'])# ,density=False)
    axs[1, 4].set_xlabel('Saccade Angle (degrees)')
    axs[1, 4].set_title('Angular Histogram of Saccades',fontweight='bold')

    # # Plot fixations locations
    # axs[1, 2].scatter(all_fixations['fix_avgpos_x'], all_fixations['fix_avgpos_y'],s=1)
    # axs[1, 2].set_xlabel('Horizontal Position [pix]')
    # axs[1, 2].set_ylabel('Vertical Position [pix]')
    # axs[1, 2].set_title('Fixations: Location',fontweight='bold')
    # axs[1, 2].set_aspect('equal')

    # Plot heatmap
    fixations_x = all_fixations['fix_avgpos_x']
    fixations_y = all_fixations['fix_avgpos_y']
    # Calculate the 2D histogram of fixations
    heatmap, xedges, yedges = np.histogram2d(fixations_y, fixations_x, bins=120)
    axs[0, 3].imshow(heatmap.T, cmap='hot', origin='lower', extent=[0, 1280, 1024, 0])#extent=[ yedges[0], yedges[-1],xedges[0], xedges[-1]])
    axs[0, 3].set_xlabel('Horizontal Position [pix]')
    axs[0, 3].set_ylabel('Vertical Position [pix]')
    axs[0, 3].set_title('Fixations: Heatmap')
    axs[0, 3].set_title('Heatmap',fontweight='bold')# Remove empty subplot

   # Plot heatmap2
    fixations_x2 = all_fixations2['fix_avgpos_x']
    fixations_y2 = all_fixations2['fix_avgpos_y']
    # Calculate the 2D histogram of fixations
    heatmap2, xedges2, yedges2 = np.histogram2d(fixations_y2, fixations_x2, bins=120)
    axs[1, 3].imshow(heatmap2.T, cmap='hot', origin='lower', extent=[0, 1280, 1024, 0])#extent=[ yedges[0], yedges[-1],xedges[0], xedges[-1]])
    axs[1, 3].set_xlabel('Horizontal Position [pix]')
    axs[1, 3].set_ylabel('Vertical Position [pix]')
    axs[1, 3].set_title('Fixations: Heatmap')
    axs[1, 3].set_title('Heatmap',fontweight='bold')# Remove empty subplot
    #fig.delaxes(axs[1, 1])

    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    #n, bins, _ = plt.hist(all_fixations['duration'], bins=100, edgecolor='black', fill=True, linewidth=0.4)

    # Find the index of the bin with the maximum frequency
    peak_index1 = np.argmax(n1) 
    peak_index2 = np.argmax(n2)

    # Get the corresponding time duration from the bin edges
    peak_duration1 = (bins1[peak_index1] + bins1[peak_index1 + 1]) / 2
    peak_duration2 = (bins2[peak_index2] + bins2[peak_index2 + 1]) / 2

    # Print the time duration at which the distribution peaks
    print(f"The distribution peaks at a duration of {peak_duration1:.2f} ms for UoN.")
    mean_duration = all_fixations['duration'].mean()
    std_duration = np.std(all_fixations['duration'])
    print(f"Mean Duration: {mean_duration:.2f} ms ± {std_duration:.2f} ms for Uon")   
    

    # Print the time duration at which the distribution peaks2
    print(f"The distribution peaks at a duration of {peak_duration2:.2f} ms for UBA.")
    mean_duration2 = all_fixations2['duration'].mean()
    std_duration2 = np.std(all_fixations2['duration'])
    print(f"Mean Duration: {mean_duration2:.2f} ms ± {std_duration2:.2f} ms for UBA")
 # Calculate the average number of fixations per trial per subject
    average_fixations_per_trial_per_subject = all_fixations.groupby(['sub_id', 'trial'])['duration'].count().groupby('sub_id').mean()
    average_fixations_per_trial = all_fixations.groupby(['sub_id', 'trial'])['duration'].count().mean()
    std_fixtr = np.std(all_fixations.groupby(['sub_id', 'trial'])['duration'].count())
    
    # Print the result
    #print(f"Average number of fixations per trial per subject: {average_fixations_per_trial:.2f} ± {std_fixtr:.2f}")
    #print(average_fixations_per_trial_per_subject)
    return fig


def plot_eye_fix_movements(all_fixations):
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 8))

    # Plot fixation distribution
    axs[0].hist(all_fixations['duration'],  bins=50, edgecolor='black', fill=True, linewidth=.4)
    #axs[0, 2].hist(saccades_tmp['duration'], bins=100, alpha=0.5, label='Fixation Durations')
    axs[0].set_xlabel('Duration [ms]')
    axs[0].set_ylabel('Cases')
    axs[0].set_title('Fixations: Duration')


    # Plot fixations locations
    axs[2].scatter(all_fixations['fix_avgpos_x'], all_fixations['fix_avgpos_y'],s=1)
    axs[2].set_xlabel('Horizontal Position [pix]')
    axs[2].set_ylabel('Vertical Position [pix]')
    axs[2].set_title('Fixations: Location')
    axs[2].set_aspect('equal')

    # Plot heatmap
    fixations_x = all_fixations['fix_avgpos_x']
    fixations_y = all_fixations['fix_avgpos_y']
    # Calculate the 2D histogram of fixations
    heatmap, xedges, yedges = np.histogram2d(fixations_y, fixations_x, bins=120)
    axs[1].imshow(heatmap, cmap='hot', origin='lower', extent=[ yedges[0], yedges[-1],xedges[0], xedges[-1]])
    axs[1].set_xlabel('Horizontal Position [pix]')
    axs[1].set_ylabel('Vertical Position [pix]')
    axs[1].set_title('Fixations: Heatmap')
    axs[1].set_title('Heatmap')# Remove empty subplot

    #fig.delaxes(axs[1, 1])
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3, top=0.4)

def plot_model_results(model_name,list_of_coeffs,figsize=[10,5],time_topos=None,top_topos=True):
    return 1


def plot_tfce_results_paper(model_name,list_of_coeffs,figsize=[10,5],time_topos=None,top_topos=True):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    import matplotlib.colors as mcolors
    from matplotlib.ticker import FuncFormatter

    vlims = [(-2.3,2.3),(-2.3,2.3),(-2.3,2.3),(-3.5,3.5),(-3.5,3.5)]
    joint_ylims = dict(eeg=[-3, 3])
    tfce_topos_ylims = [dict(eeg=[-2.5, 2.5]),dict(eeg=[-2.5, 2.5]),dict(eeg=[-2.5, 2.5]),dict(eeg=[-2.5, 2.5]),dict(eeg=[-2.5, 2.5])]
    top_slide = 0.02
    horizontal_jump = 0.2
    
    
    
    if time_topos is not None:
        time_plots = time_topos
    
        
    combined_model_path = paths().combined_model_path(model_name)
    coeffs = list_of_coeffs
    fig = plt.figure(constrained_layout=False,figsize=figsize) 

    jump=0
    for coeff in list_of_coeffs:
        if top_topos:
            ax_topo1 = fig.add_axes((0.033+jump*horizontal_jump, 0.75, 0.038, 0.07))
            ax_topo2 = fig.add_axes((0.033+jump*horizontal_jump+top_slide+0.03, 0.75, 0.038, 0.07))
            ax_topo3 = fig.add_axes((0.033+jump*horizontal_jump+top_slide*2+0.03*2, 0.75, 0.038, 0.07))
            ax_topo_cb = fig.add_axes((0.033+jump*horizontal_jump+top_slide*3+0.03*3, 0.75, 0.004, 0.09))
            axs_topos = [ax_topo1, ax_topo2, ax_topo3, ax_topo_cb]
        # 
        ax_frp = fig.add_axes((0.033+jump*horizontal_jump, 0.47, 0.152, 0.2))
        ax_tfce = fig.add_axes((0.033+jump*horizontal_jump, 0.27, 0.152, 0.2))
        ax_tfce_topo = fig.add_axes((0.08+jump*horizontal_jump, -0.015, 0.06, 0.25))
        ax_tfce_topo_cb = fig.add_axes((0.2+jump*horizontal_jump, 0.08, 0.003, 0.3))
        jump+=1
        # Group axes

        # file names
        ave_fname = f'/grand_average_coeff_{coeff}-ave.fif'
        clus_fname = f'/cluster_{coeff}.npy'
        
        
        # create gridspec for topos and main plot of rFRPs
        grand_avg = mne.read_evokeds(combined_model_path+ave_fname, baseline=(None, 0), verbose=False)
        grand_avg[0].nave = None
        # grand_avg[0].ch_type = None

        if top_topos:
            grand_avg[0].plot_joint(title="",ts_args={'xlim': (-.1,.4),'ylim':joint_ylims,'axes':ax_frp,'titles':dict(eeg=''),'window_title':''},
                                        topomap_args={'vlim':vlims[coeff],'contours':2,'axes':axs_topos,'size':.8},
                                        show=False)
        else:
            if coeff ==2:
                xmax = .85
            else:
                xmax = .85
            grand_avg[0].plot(axes=ax_frp,titles=dict(eeg=''),window_title='',xlim= (-.1,xmax),ylim=joint_ylims,
                            show=False)
            
        time_plot = time_plots[coeff] # For highlighting a specific time. 
        ax_frp.axvline(time_plot, ls="--", color="k", lw=1)
        ax_frp.axvline(0, ls="-", color="k", lw=.9)

        # clean axis
        ax_frp.set_xlabel([])
        ax_frp.set_xticklabels([])
        ax_frp.set_title('')
        if top_topos:
            ax_cb = axs_topos[-1]
            ax_cb.set_title(f'$\mu V$')
                # topos fonts
            for top in axs_topos:
                top.title.set_fontsize(10)
                # top.set_title('')

        fig.set_label('')
        fig.legends = []
        ax_tfce.legend().set_visible(False)



        # TFCE Plot
        clusters_mask = np.load(combined_model_path+clus_fname)
        ax_tfce.axvline(time_plot, ls="--", color="k", lw=1)
        ax_tfce.axvline(0, ls="-", color="k", lw=.9)
        # Greys
        greys_cmap = plt.cm.get_cmap('Greys')
        colors = greys_cmap(np.linspace(0, 1, 256))
        # Adjust the luminance values to make the colormap darker
        colors[:, :3] *= 0.6  # Multiply RGB values by 0.7 to darken them
        custom_cmap= mcolors.ListedColormap(colors)
        #title = 'TFCE p-value'# with alpha level={pval_threshold}'
        if coeff ==2:
            xmax = .85
            grand_avg[0].plot_image(cmap='RdBu_r', mask=clusters_mask, mask_style='mask', mask_alpha=0.5,
                            titles=None, axes=ax_tfce,show=False,xlim=(-.1,xmax),mask_cmap=custom_cmap, clim=tfce_topos_ylims[coeff],colorbar=False)
    
        else:
            xmax = .85
            grand_avg[0].plot_image(cmap='RdBu_r', mask=clusters_mask, mask_style='mask', mask_alpha=0.5,
                            titles=None, axes=ax_tfce,show=False,xlim=(-.1,xmax),mask_cmap=custom_cmap, clim=tfce_topos_ylims[coeff],colorbar=False)
    
        # clean tfce axis
        ax_tfce.set_title('') 
        if jump!=1:
            #clean all but the first
            ax_frp.set_yticklabels([])
            ax_tfce.set_yticklabels([])
            ax_tfce.set_ylabel('')
            ax_frp.set_ylabel('')

        times = grand_avg[0].times
        ix_plot = np.argmin(np.abs(time_plot - times))
        
        coeff_data = grand_avg[0].get_data()
        max_coef = coeff_data.max()
        vlims_tfce_topo=(tfce_topos_ylims[coeff]['eeg'][0]*1e-6,tfce_topos_ylims[coeff]['eeg'][1]*1e-6)
        topo,cm = mne.viz.plot_topomap(
            coeff_data[:, ix_plot], pos=grand_avg[0].info, axes=ax_tfce_topo, show=False, vlim=(vlims_tfce_topo[0],vlims_tfce_topo[1]),#vlim=(-max_coef, max_coef)
        mask=clusters_mask[:,ix_plot],mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=3),contours=4,cmap='RdBu_r')
        v1 = np.linspace(vlims_tfce_topo[0], vlims_tfce_topo[1], 5, endpoint=True)
        clb = fig.colorbar(topo, cax=ax_tfce_topo_cb,ticks=v1)
        # Define the formatting function
        def format_ticks(value, pos):
            return f'{value * 1e6:.1f}'
        clb.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
        clb.ax.yaxis.set_tick_params(labelsize=7)  # Adjust font size here

        ax_tfce_topo_cb.set_title(f'$\mu V$',fontsize=12) # title
        ax_tfce_topo.set_xlabel("%s ms" % time_plot)
        ax_tfce_topo.title.set_size(10)

        ax_frp.set_title('')
    ch = fig.get_children()
    for ax in ch:
        childs = ax.get_children()
        for c in childs:
            if isinstance(c,plt.Text):
                if '(64' in c.get_text():
                    c.remove()
    # plt.legend('')
    # child = fig.get_children()
    # ax_erase = child[10].remove()
    return fig


def epochs(subject, epochs, picks, order=None, overlay=None, combine='mean', sigma=5, group_by=None, cmap='jet',
           vmin=None, vmax=None, display_figs=True, save_fig=None, fig_path=None, fname=None):

    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    fig_ep = epochs.plot_image(picks=picks, order=order, sigma=sigma, cmap=cmap, overlay_times=overlay, combine=combine,
                               vmin=vmin, vmax=vmax, title=subject.subject_id, show=display_figs)

    # Save figure
    if save_fig:
        if len(fig_ep) == 1:
            fig = fig_ep[0]
            functions.fig(fig=fig, path=fig_path, fname=fname)
        else:
            for i in range(len(fig_ep)):
                fig = fig_ep[i]
                group = group_by.keys()[i]
                fname += f'{group}'
                save.fig(fig=fig, path=fig_path, fname=fname)


def evoked(evoked_eeg, evoked_misc, picks, plot_gaze=False, fig=None,
           axes=None, plot_xlim='tight', plot_ylim=None, display_figs=False, save_fig=True, fig_path=None, fname=None):
    '''
    Plot evoked response with mne.Evoked.plot() method. Option to plot gaze data on subplot.

    :param evoked_meg: Evoked with picked mag channels
    :param evoked_misc: Evoked with picked misc channels (if plot_gaze = True)
    :param picks: Meg channels to plot
    :param plot_gaze: Bool
    :param fig: Optional. Figure instance
    :param axes: Optional. Axes instance (if figure provided)
    :param plot_xlim: tuple. x limits for evoked and gaze plot
    :param plot_ylim: dict. Possible keys: meg, mag, eeg, misc...
    :param display_figs: bool. Whether to show figures or not
    :param save_fig: bool. Whether to save figures. Must provide save_path and figure name.
    :param fig_path: string. Optional. Path to save figure if save_fig true
    :param fname: string. Optional. Filename if save_fig is True

    :return: None
    '''

    # Sanity check
    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if axes:
        if save_fig and not fig:
            raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

        evoked_meg.plot(picks=picks, gfp=True, axes=axes, time_unit='s', spatial_colors=True, xlim=plot_xlim,
                        ylim=plot_ylim, show=display_figs)
        axes.vlines(x=0, ymin=axes.get_ylim()[0], ymax=axes.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    elif plot_gaze:
        # Get Gaze x ch
        gaze_x_ch_idx = np.where(np.array(evoked_misc.ch_names) == 'ET_gaze_x')[0][0]
        fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        axs[1].plot(evoked_misc.times, evoked_misc.data[gaze_x_ch_idx, :])
        axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
        axs[1].set_ylabel('Gaze x')
        axs[1].set_xlabel('Time')

        evoked_meg.plot(picks=picks, gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim,
                        ylim=plot_ylim, show=display_figs)
        axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    else:
        fig = evoked_eeg.plot(picks=picks, gfp=True, axes=axes, time_unit='s', spatial_colors=True, xlim=plot_xlim,
                              ylim=plot_ylim, show=display_figs)
        axes = fig.get_axes()[0]
        axes.vlines(x=0, ymin=axes.get_ylim()[0], ymax=axes.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            functions.fig(fig=fig, path=fig_path, fname=fname)


def evoked_topo(evoked_meg, picks, topo_times, title=None, fig=None, axes_ev=None, axes_topo=None, xlim=None, ylim=None,
                display_figs=False, save_fig=False, fig_path=None, fname=None):

    # Sanity check
    if save_fig and (not fig_path or not fname):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if axes_ev and axes_topo:
        if save_fig and not fig:
            raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

        display_figs = True

        evoked_meg.plot_joint(times=topo_times, title=title, picks=picks, show=display_figs,
                              ts_args={'axes': axes_ev, 'xlim': xlim, 'ylim': ylim},
                              topomap_args={'axes': axes_topo})

        axes_ev.vlines(x=0, ymin=axes_ev.get_ylim()[0], ymax=axes_ev.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    else:
        fig = evoked_meg.plot_joint(times=topo_times, title=title, picks=picks, show=display_figs,
                                    ts_args={'xlim': xlim, 'ylim': ylim})

        all_axes = plt.gcf().get_axes()
        axes_ev = all_axes[0]
        axes_ev.vlines(x=0, ymin=axes_ev.get_ylim()[0], ymax=axes_ev.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

def fig_versus():

    fig = plt.figure(figsize=(6, 9))

    # 1st row Topoplots
    #MS
    ax4 = fig.add_axes([0.2, 0.88, 0.1, 0.1])
    ax5 = fig.add_axes([0.34, 0.88, 0.013, 0.1])

    # 2nd row Evokeds 
    ax7 = fig.add_axes([0.1, 0.71, 0.6, 0.15])

    # 3 row Topoplots
    # VS
 
    # topo
    ax10 = fig.add_axes([0.2, 0.54, 0.1, 0.1])
    ax11 = fig.add_axes([0.34, 0.54, 0.013, 0.1])

    # 4 row Evokeds
    ax13 = fig.add_axes([0.1, 0.38, 0.6, 0.15])

    # 5 row Topoplot Difference
    ax14 = fig.add_axes([0.2, 0.22, 0.1, 0.1]) #topo plot diff
    ax15 = fig.add_axes([0.34, 0.22, 0.013, 0.1])

    # 6th row Evoked Diference
    ax16 = fig.add_axes([0.1, 0.05, 0.6, 0.15])

    # groups

    ax_evoked_cond_1 = ax7
    ax_topo_cond_1 = [ax4, ax5]


    ax_evoked_cond_2 = ax13
    ax_topo_cond_2 = [ax10, ax11]

    ax_evoked_diff = ax16
    ax_topo_diff = [ax14, ax15]

    return fig, ax_evoked_cond_2, ax_evoked_cond_1, \
           ax_topo_cond_1, ax_topo_cond_2, ax_evoked_diff, ax_topo_diff

def plot_et_cond(phase=None,present=None,mss=None,correct=None,item_type=None,expepiment=None,uniform_back=None):

    info = exp_info(expepiment)
    metadata_path = paths(expepiment).full_metadata_path()
    subjects_ids = info.subjects_ids
    rej_subjects_ids = info.rejected_subjects

    #----------parameters----------#
    all_fixations = pd.DataFrame()  # Initialize an empty DataFrame
    all_saccades = pd.DataFrame()
    filtered_subjects_ids = [sujs for sujs in subjects_ids if sujs not in rej_subjects_ids]
    for sub_id in filtered_subjects_ids:
        eventos = pd.read_csv(os.path.join(metadata_path,f'{sub_id}_full_metadata.csv'))
        #eventos = eventos.loc[(eventos.bad == 0) & (eventos.inrange == True) ]
        #mod 2024 to exclude backgrounds not being 1024x1280
        eventos['sub_id'] = sub_id  # Add 'sub_id' column
        blacklist = []
        if uniform_back:
            suj = load.subject(info,sub_id)

            path = paths()
            exp_path = path.experiment_path()

            bh_data = suj.load_bh_csv()
            image_names = bh_data['searchimage'].drop_duplicates()
            image_names = image_names.str.split('cmp_', expand=True)[1]
            image_names = image_names.str.split('.jpg', expand=True)[0]
            image_names = list(image_names)
            for idx,image_name in enumerate(image_names[:-1]):
                img = plt.imread(exp_path + 'cmp_' + image_name + '.jpg')    
                if img.shape != (1024, 1280, 3):
                    blacklist.append(idx+1)




        eventos = eventos.loc[~eventos['trial'].isin(blacklist)]
        if correct:
            eventos = eventos.loc[(eventos.correct == correct)]
        if present:
            eventos = eventos.loc[(eventos.present == present)]
        if mss:
            eventos = eventos.loc[(eventos.mss == mss)]
        fixations_tmp = eventos[(eventos['type']=='fixation') & (eventos['phase']==phase)]
        if item_type == 'ontarget':
            fixations_tmp = fixations_tmp.loc[(fixations_tmp['ontarget'] == True)]
        elif item_type == 'ondistractor':
            fixations_tmp = fixations_tmp.loc[(fixations_tmp['ondistractor'] == True)]
        saccades_tmp = eventos[(eventos['type']=='saccade') & (eventos['phase']==phase)]
        all_fixations = pd.concat([all_fixations, fixations_tmp], ignore_index=True)
        all_saccades  = pd.concat([all_saccades, saccades_tmp], ignore_index=True)
        
    print(all_fixations.shape)
    fig = plot_eye_movements(all_fixations,all_saccades)
    plt.show()
    return fig

def data_to_plot_et_cond(phase=None,present=None,mss=None,correct=None,item_type=None,expepiment=None,uniform_back=None,exclude_first=None,dur_min=None):

    info = exp_info(expepiment)
    metadata_path = paths(expepiment).full_metadata_path()
    subjects_ids = info.subjects_ids
    rej_subjects_ids = info.rejected_subjects

    #----------parameters----------#
    all_fixations = pd.DataFrame()  # Initialize an empty DataFrame
    all_saccades = pd.DataFrame()
    filtered_subjects_ids = [sujs for sujs in subjects_ids if sujs not in rej_subjects_ids]
    for sub_id in filtered_subjects_ids:
        eventos = pd.read_csv(os.path.join(metadata_path,f'{sub_id}_full_metadata.csv'))
        #eventos = eventos.loc[(eventos.bad == 0) & (eventos.inrange == True) ]
        #mod 2024 to exclude backgrounds not being 1024x1280
        eventos['sub_id'] = sub_id  # Add 'sub_id' column
        blacklist = []
        if uniform_back:
            suj = load.subject(info,sub_id)

            path = paths()
            exp_path = path.experiment_path()

            bh_data = suj.load_bh_csv()
            image_names = bh_data['searchimage'].drop_duplicates()
            image_names = image_names.str.split('cmp_', expand=True)[1]
            image_names = image_names.str.split('.jpg', expand=True)[0]
            image_names = list(image_names)
            for idx,image_name in enumerate(image_names[:-1]):
                img = plt.imread(exp_path + 'cmp_' + image_name + '.jpg')    
                if img.shape != (1024, 1280, 3):
                    blacklist.append(idx+1)



            
        eventos = eventos.loc[~eventos['trial'].isin(blacklist)]
        if correct:
            eventos = eventos.loc[(eventos.correct == correct)]
        if present:
            eventos = eventos.loc[(eventos.present == present)]
        if mss:
            eventos = eventos.loc[(eventos.mss == mss)]
        if exclude_first:
            eventos = eventos.loc[(eventos['rank'] != 1)]


        fixations_tmp = eventos[(eventos['type']=='fixation') & (eventos['phase']==phase)]
        if item_type == 'ontarget':
            fixations_tmp = fixations_tmp.loc[(fixations_tmp['ontarget'] == True)]
        elif item_type == 'ondistractor':
            fixations_tmp = fixations_tmp.loc[(fixations_tmp['ondistractor'] == True)]
        saccades_tmp = eventos[(eventos['type']=='saccade') & (eventos['phase']==phase)]
        all_fixations = pd.concat([all_fixations, fixations_tmp], ignore_index=True)
        if dur_min:
            all_fixations = all_fixations.loc[all_fixations.duration>dur_min]

        all_saccades  = pd.concat([all_saccades, saccades_tmp], ignore_index=True)
        
    print(all_fixations.shape)
    return all_fixations, all_saccades
    

def plot_et_fix_cond(phase=None,present=None,mss=None,correct=None,item_type=None,subject_id=None):                                                                                            
    info = exp_info()
    metadata_path = paths().full_metadata_path()
    subjects_ids = info.subjects_ids
    #----------parameters-------------
    all_fixations = pd.DataFrame()  # Initialize an empty DataFrame

    if subject_id != None :
        subjects_ids = [subject_id]
 
    for sub_id in subjects_ids:
        eventos = pd.read_csv(os.path.join(metadata_path,f'{sub_id}_full_metadata.csv'))
        #eventos = eventos.loc[(eventos.bad == 0) & (eventos.inrange == True) ]
        if correct is not None:
            eventos = eventos.loc[(eventos.correct == correct)]
        if present is not None:
            eventos = eventos.loc[(eventos.present == present)]
        if mss:
            eventos = eventos.loc[(eventos.mss == mss)]
        fixations_tmp = eventos.loc[(eventos['type']=='fixation') & (eventos['phase']==phase)]
        if item_type == 'ontarget':
            fixations_tmp = fixations_tmp.loc[(fixations_tmp['ontarget'] == True)]
        elif item_type == 'ondistractor':
            fixations_tmp = fixations_tmp.loc[(fixations_tmp['ondistractor'] == True)]
        saccades_tmp = eventos[(eventos['type']=='saccade') & (eventos['phase']==phase)]
        all_fixations = pd.concat([all_fixations, fixations_tmp], ignore_index=True)
        
    print(all_fixations.shape)
    plot_eye_fix_movements(all_fixations)
    plt.show()

def plot_vif_distriburion(model_path_1,model_path_2):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    ll_fname = f'/intercept_vif_aves.npy'

    # Sample data (replace with your actual VIF coefficients)
    ll_data_1 = np.load(model_path_1 + ll_fname)  # VIF coefficients for dataset 1
    ll_data_2 = np.load(model_path_2 + ll_fname)  # VIF coefficients for dataset 2

    # Combine VIF coefficients from both datasets
    print(ll_data_1.shape)
    print(ll_data_2.shape)
    combined_ll_data = np.vstack((ll_data_1, ll_data_2))
    # Generate labels for datasets

    ll_data = combined_ll_data
    column_names= []
    for i in range(ll_data.shape[1]):
        column_names.append(f'coeff_{i}')
    VIF_df = pd.DataFrame(ll_data, columns=column_names)
    # Create violin plot using seaborn with rotated axes
    ax = sns.violinplot(data=VIF_df, orient='h',linewidth=0.5)
    plt.title('Distribution of VIF for Each Coefficient')
    plt.xlabel('VIF')
    plt.ylabel('Coefficient')
    num_samples = len(VIF_df)
    plt.text(0.95, 0.95, f'N = {num_samples}', ha='right', va='top', transform=ax.transAxes,
             fontsize=10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
    

    plt.show()


def plot_llaicbiccp_ave(model_path_1,model_path_2):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pdb
    ll_fname = f'/ll_aic_bic_cp.npy'

    # Sample data (replace with your actual VIF coefficients)
    ll_data_1 = np.load(model_path_1 + ll_fname)  # VIF coefficients for dataset 1
    ll_data_2 = np.load(model_path_2 + ll_fname)  # VIF coefficients for dataset 2
    # pdb.set_trace()

    # Combine VIF coefficients from both datasets
    print(ll_data_1.shape)
    print(ll_data_2.shape)
    combined_ll_data = np.concatenate([ll_data_1, ll_data_2],axis=0)
    # Generate labels for datasets

    ll_data = combined_ll_data
    column_names= ['AIC','BIC']
    ll_df = pd.DataFrame(np.mean(ll_data,axis=1), columns=column_names)
    # Create violin plot using seaborn with rotated axes
    ax = sns.violinplot(data=ll_df, orient='h',linewidth=0.5)
    plt.title('AIC BIC')
    plt.xlabel('ll/Cp')
    plt.ylabel('Criteria')
    num_samples = len(ll_df)
    plt.text(0.95, 0.95, f'N = {num_samples}', ha='right', va='top', transform=ax.transAxes,
             fontsize=10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
    

    plt.show()




def fig_psd():

    fig = plt.figure(figsize=(15, 5))

    # 1st row Topoplots
    ax1 = fig.add_axes([0.05, 0.6, 0.15, 0.3])
    ax2 = fig.add_axes([0.225, 0.6, 0.15, 0.3])
    ax3 = fig.add_axes([0.4, 0.6, 0.15, 0.3])
    ax4 = fig.add_axes([0.575, 0.6, 0.15, 0.3])
    ax5 = fig.add_axes([0.75,  0.6, 0.15, 0.3])

    # 2nd row PSD
    ax6 = fig.add_axes([0.15,  0.1, 0.7, 0.4])

    # Group axes
    axs_topo = [ax1, ax2, ax3, ax4, ax5]
    ax_psd = ax6

    return fig, axs_topo, ax_psd


def fig_time_frequency(fontsize=None, ticksize=None):

    if fontsize:
        matplotlib.rc({'font.size': fontsize})
    if ticksize:
        matplotlib.rc({'xtick.labelsize': ticksize})
        matplotlib.rc({'ytick.labelsize': ticksize})

    fig, axes_topo = plt.subplots(3, 3, figsize=(15, 8), gridspec_kw={'width_ratios': [5, 1, 1]})

    for ax in axes_topo[:, 0]:
        ax.remove()
    axes_topo = [ax for ax_arr in axes_topo[:, 1:] for ax in ax_arr ]

    ax1 = fig.add_axes([0.1, 0.1, 0.5, 0.8])

    return fig, axes_topo, ax1


def tfr(tfr, chs_id, epoch_id, mss, cross1_dur, mss_duration, cross2_dur, plot_xlim=(None, None), baseline=None, bline_mode=None,
        dB=False, vmin=None, vmax=None, subject=None, title=None, topo_times=None, display_figs=False, save_fig=False, fig_path=None, fname=None,
        fontsize=None, ticksize=None):

    # Sanity check
    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if plot_xlim == (None, None):
        plot_xlim = (tfr.tmin, tfr.tmax)

    # Turn off dB if baseline mode is incompatible with taking log10
    if dB and bline_mode in ['mean', 'logratio']:
        dB = False

    # Define figure
    fig, axes_topo, ax_tf = fig_time_frequency(fontsize=fontsize, ticksize=ticksize)

    # Pick plot channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=tfr.info)

    # Plot time-frequency
    tfr.plot(picks=picks, baseline=baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1],
             combine='mean', cmap='jet', axes=ax_tf, show=display_figs, vmin=vmin, vmax=vmax, dB=dB)

    # Plot time markers as vertical lines
    if 'ms' in epoch_id and mss:
        ax_tf.vlines(x=0, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
        try:
            ax_tf.vlines(x=mss_duration[mss], ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1],
                         linestyles='--', colors='black')
        except: pass
        try:
            ax_tf.vlines(x=mss_duration[mss] + cross2_dur, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1],
                         linestyles='--', colors='black')
        except: pass

    elif 'cross2' in epoch_id:
        try:
            ax_tf.vlines(x=cross2_dur, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
        except: pass

    else:
        ax_tf.vlines(x=0, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')

    # Topomaps parameters
    if not topo_times:
        topo_times = plot_xlim
    topomap_kw = dict(ch_type='mag', tmin=topo_times[0], tmax=topo_times[1], baseline=baseline,
                      mode=bline_mode, show=display_figs)
    plot_dict = dict(Delta=dict(fmin=1, fmax=4), Theta=dict(fmin=4, fmax=8), Alpha=dict(fmin=8, fmax=12),
                     Beta=dict(fmin=12, fmax=30), Gamma=dict(fmin=30, fmax=45), HGamma=dict(fmin=45, fmax=100))

    # Plot topomaps
    for ax, (title_topo, fmin_fmax) in zip(axes_topo, plot_dict.items()):
        try:
            tfr.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
        except:
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title_topo)

    # Figure title
    if title:
        fig.suptitle(title)
    elif subject:
        fig.suptitle(subject.subject_id + f'_{fname.split("_")[0]}_{chs_id}_{bline_mode}_topotimes_{topo_times}')
    elif not subject:
        fig.suptitle(f'Grand_average_{fname.split("_")[0]}_{chs_id}_{bline_mode}_topotimes_{topo_times}')
        fname = 'GA_' + fname

    fig.tight_layout()

    if save_fig:
        fname += f'_topotimes_{topo_times}'
        os.makedirs(fig_path, exist_ok=True)
        save.fig(fig=fig, path=fig_path, fname=fname)


def tfr_plotjoint(tfr, plot_baseline=None, bline_mode=None, plot_xlim=(None, None), plot_max=True, plot_min=True, vlines_times=[0],
                  vmin=None, vmax=None, display_figs=False, save_fig=False, trf_fig_path=None, fname=None, fontsize=None, ticksize=None):
    # Sanity check
    if save_fig and (not fname or not trf_fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if fontsize:
        matplotlib.rc({'font.size': fontsize})
    if ticksize:
        matplotlib.rc({'xtick.labelsize': ticksize})
        matplotlib.rc({'ytick.labelsize': ticksize})

    if plot_baseline:
        tfr_plotjoint = tfr.copy().apply_baseline(baseline=plot_baseline, mode=bline_mode)
    else:
        tfr_plotjoint = tfr.copy()

    # Get all mag channels to plot
    picks = functions_general.pick_chs(chs_id='mag', info=tfr.info)
    tfr_plotjoint = tfr_plotjoint.pick(picks)

    # Get maximum
    timefreqs = functions_analysis.get_plot_tf(tfr=tfr_plotjoint, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min)

    # Title
    if fname:
        title = f'{fname.split("_")[1]}_{bline_mode}'
    else:
        f'{bline_mode}'

    fig = tfr_plotjoint.plot_joint(timefreqs=timefreqs, tmin=plot_xlim[0], tmax=plot_xlim[1], cmap='jet', vmin=vmin, vmax=vmax,
                                   title=title, show=display_figs)

    # Plot vertical lines
    tf_ax = fig.axes[0]
    for t in vlines_times:
        try:
            tf_ax.vlines(x=t, ymin=tf_ax.get_ylim()[0], ymax=tf_ax.get_ylim()[1], linestyles='--', colors='gray')
        except:
            pass

    if save_fig:
        save.fig(fig=fig, path=trf_fig_path, fname=fname)


def tfr_plotjoint_picks(tfr, plot_baseline=None, bline_mode=None, plot_xlim=(None, None), plot_max=True, plot_min=True,
                        vmin=None, vmax=None, chs_id='mag', vlines_times=[0],
                        display_figs=False, save_fig=False, trf_fig_path=None, fname=None, fontsize=None, ticksize=None):
    # Sanity check
    if save_fig and (not fname or not trf_fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if fontsize:
        matplotlib.rc({'font.size': fontsize})
    if ticksize:
        matplotlib.rc({'xtick.labelsize': ticksize})
        matplotlib.rc({'ytick.labelsize': ticksize})

    if plot_baseline:
        tfr_plotjoint = tfr.copy().apply_baseline(baseline=plot_baseline, mode=bline_mode)
        tfr_topo = tfr.copy().apply_baseline(baseline=plot_baseline, mode=bline_mode)  # tfr for topoplots
    else:
        tfr_plotjoint = tfr.copy()
        tfr_topo = tfr.copy()  # tfr for topoplots

    # TFR from certain chs and topoplots from all channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=tfr.info)
    tfr_plotjoint = tfr_plotjoint.pick(picks)

    # Get maximum
    timefreqs = functions_analysis.get_plot_tf(tfr=tfr_plotjoint, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min)

    # Title
    if fname:
        title = f'{fname.split("_")[1]}_{bline_mode}'
    else:
        f'{bline_mode}'

    # Plot tf plot joint
    fig = tfr_plotjoint.plot_joint(timefreqs=timefreqs, tmin=plot_xlim[0], tmax=plot_xlim[1], cmap='jet', vmin=vmin, vmax=vmax,
                                   title=title, show=display_figs)

    # Plot vertical lines
    tf_ax = fig.axes[0]
    for t in vlines_times:
        try:
            tf_ax.vlines(x=t, ymin=tf_ax.get_ylim()[0], ymax=tf_ax.get_ylim()[1], linestyles='--', colors='gray')
        except:
            pass

    # Get min and max from all topoplots
    maxs = []
    for timefreq in timefreqs:
        data = tfr_topo.copy().crop(tmin=timefreq[0], tmax=timefreq[0], fmin=timefreq[1], fmax=timefreq[1]).data.ravel()
        maxs.append(np.abs(data).max())
    vmax = np.max(maxs)

    # Get topo axes and overwrite
    topo_axes = fig.axes[1:-1]
    for ax, timefreq in zip(topo_axes, timefreqs):
        fmin_fmax = dict(fmin=timefreq[1], fmax=timefreq[1])
        topomap_kw = dict(ch_type='mag', tmin=timefreq[0], tmax=timefreq[0], colorbar=False, show=display_figs)
        tfr_topo.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw, cmap='jet', vlim=(-vmax, vmax))

    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='jet')
    # Get colorbar axis
    cbar_ax = fig.axes[-1]
    fig.colorbar(sm, cax=cbar_ax)

    if save_fig:
        save.fig(fig=fig, path=trf_fig_path, fname=fname)


def plot_categorical_balance(model_path_1, model_path_2, use_splines=False):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    fname = f'/matrix_balance.npy'
    print(use_splines)
    # Sample data (replace with your actual VIF coefficients)
    data_1 = np.load(model_path_1 + fname)  # VIF coefficients for dataset 1
    data_2 = np.load(model_path_2 + fname)  # VIF coefficients for dataset 2
    # import pdb
    # pdb.set_trace()
    # Combine VIF coefficients from both datasets
    print(data_1.shape)
    print(data_2.shape)
    combined_data = np.concatenate((data_1, data_2),axis=0)

    # Generate labels for datasets
    ncols = combined_data.shape[1]
    column_names= []
    if use_splines:
        sacc_id = ncols-5
    else:
        sacc_id = ncols-1
    for i in range(ncols):
        if i==0:
            column_names.append('fix_int')
        elif i==sacc_id:
            column_names.append('sacc_int')
        else:
            column_names.append(f'coeff_{i}')


    x = column_names


    # Create DataFrame
    VIF_df = pd.DataFrame(combined_data, columns=column_names)

    # Calculate mean number of instances for each coefficient
    mean_instances = VIF_df.mean()

    # Create violin plot using seaborn with rotated axes
    ax = sns.violinplot(data=VIF_df, orient='h',linewidth=0.5)
    plt.title('Distribution of N for Each Coefficient')
    plt.xlabel('N')
    plt.ylabel('Coefficient')
    offset = .2
    # Annotate the plot with mean values
    for i, mean_instance in enumerate(mean_instances):
        plt.text(mean_instance+.05, i-offset, f'Mean: {mean_instance:.2f}', color='black', va='center')

    plt.show()
    # plt.figure(figsize=(4,3))

    # # stack bars
    # y_bottom = 0
    # for i,coeff in enumerate(range(ncols)):
    #     plt.bar(x[i], combined_data[:,coeff].mean(), bottom=y_bottom,label=coeff)

    # for i,xs in enumerate(x):
    #     plt.text(xs, combined_data[:,i].mean()/2, "%.1f"%combined_data[:,i].mean(), ha="center", va="center")
    #     # for xpos, ypos, yval in zip(x, y1+y2/2, y2):
    #     #     plt.text(xpos, ypos, "%.1f"%yval, ha="center", va="center")
    #     # for xpos, ypos, yval in zip(x, y1+y2+y3/2, y3):
    #     #     plt.text(xpos, ypos, "%.1f"%yval, ha="center", va="center")

    # # # add text annotation corresponding to the "total" value of each bar
    # # for xpos, ypos, yval in zip(x, fix_coeff, n_fix_coeff):
    # #     plt.text(xpos, ypos, "N=%d"%yval, ha="center", va="bottom")

    # # plt.ylim(0,110)

    # plt.legend(bbox_to_anchor=(1.01,0.5), loc='center left')


    # plt.show()

if __name__=='__main__':
    plot_et_cond(phase='vs',present=None,mss=None,correct=None)
    
    