from pydeconv.pydeconv import PyDeconv 
from pydeconv.pydeconv_functions import analyze_data
import pandas as pd

data_path = "/example_data/"

settings = analyze_data()
features = pd.read_csv(data_path+"629959_full_metadata.csv") 
# Initialize the model
rERP_model = PyDeconv(settings = settings , features = features)
X_design = rERP_model.create_matrix()
y_data   = rERP_model.get_data()



# Prepare model data (make time the first dimension)
speech = speech.T
Y, _ = raw[:]  # Outputs for the model
Y = Y.T

# Iterate through splits, fit the model, and predict/test on held-out data
coefs = np.zeros((n_splits, n_channels, n_delays))
scores = np.zeros((n_splits, n_channels))
for ii, (train, test) in enumerate(cv.split(speech)):
    print(f"split {ii + 1} / {n_splits}")
    rf.fit(speech[train], Y[train])
    scores[ii] = rf.score(speech[test], Y[test])
    # coef_ is shape (n_outputs, n_features, n_delays). we only have 1 feature
    coefs[ii] = rf.coef_[:, 0, :]
times = rf.delays_ / float(rf.sfreq)

# Average scores and coefficients across CV splits
mean_coefs = coefs.mean(axis=0)
mean_scores = scores.mean(axis=0)

# Plot mean prediction scores across all channels
fig, ax = plt.subplots(layout="constrained")
ix_chs = np.arange(n_channels)
ax.plot(ix_chs, mean_scores)
ax.axhline(0, ls="--", color="r")
ax.set(title="Mean prediction score", xlabel="Channel", ylabel="Score ($r$)")