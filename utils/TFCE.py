
import numpy as np
import mne 
'''100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Permuting : 255/255 [05:56<00:00,    1.40s/it]
Traceback (most recent call last):
  File "/home/dac/Documents/Repos/EEGDECONV/ga_unfold.py", line 62, in <module>
    clusters_mask, pval_threshold = tfce(evokeds_data,ch_adjacency_sparse)
  File "/home/dac/Documents/Repos/EEGDECONV/TFCE.py", line 28, in tfce
    p_tfce = p_tfce.reshape(len(evoked.times), len(evoked.ch_names)).T
NameError: name 'evoked' is not defined'''
def tfce(observations,ch_adjacency_sparse, n_permutations=512, alpha = 0.05):

    # Permutation cluster test parameters
    degrees_of_freedom = observations.shape[0]- 1
    # t_thresh = scipy.stats.t.ppf(1 - desired_pval / 2, df=degrees_of_freedom)
    t_thresh = dict(start=0, step=0.2)
    # Get channel adjacency
    # Clusters out type
    if type(t_thresh) == dict:
        out_type = 'indices'
    else:
        out_type = 'mask'

    # Permutations cluster test (TFCE if t_thresh as dict)
    t_tfce, clusters, p_tfce, H0 = mne.stats.permutation_cluster_1samp_test(X=observations, 
                                                                    threshold=t_thresh,
                                                                    adjacency=ch_adjacency_sparse,
                                                                    n_permutations=n_permutations, 
                                                                    out_type=out_type, 
                                                                    n_jobs=-1)

    pval_threshold = alpha
    # Make clusters mask
    if type(t_thresh) == dict:
        # If TFCE use p-vaues of voxels directly
        p_tfce = p_tfce.reshape( observations.shape[1], observations.shape[2]).T

        # Reshape to data's shape
        clusters_mask = p_tfce < pval_threshold
    else:
        # Get significant clusters
        good_clusters_idx = np.where(p_tfce < pval_threshold)[0]
        significant_clusters = [clusters[idx] for idx in good_clusters_idx]

        # Rehsape to data's shape by adding all clusters into one bool array
        clusters_mask = np.zeros(clusters[0].shape)
        for significant_cluster in significant_clusters:
            clusters_mask += significant_cluster
        clusters_mask = clusters_mask.astype(bool).T

    return clusters_mask, pval_threshold


def get_channel_adjacency(info):

    # Compute channel adjacency from montage info
    ch_adjacency = mne.channels.find_ch_adjacency(info=info, ch_type=None)

    # Default ctf275 info has 275 channels, we are using 271. Check for extra channels
    extra_chs_idx = [i for i, ch in enumerate(ch_adjacency[1]) if ch not in info.ch_names]

    if len(extra_chs_idx):
        ch_adjacency_mat = ch_adjacency[0].toarray()

        # Remove extra channels
        for extra_ch in extra_chs_idx:
            ch_adjacency_mat = np.delete(ch_adjacency_mat, extra_ch, axis=0)
            ch_adjacency_mat = np.delete(ch_adjacency_mat, extra_ch, axis=1)

        # Reformat to scipy sparce matrix
        ch_adjacency_sparse = scipy.sparse.csr_matrix(ch_adjacency_mat)

    else:
        ch_adjacency_sparse = ch_adjacency[0]

    return ch_adjacency_sparse