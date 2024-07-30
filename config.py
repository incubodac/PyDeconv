# config.py

events_of_interest = {
    "first_intercept_event_type": "fixation",
    "second_intercept_event_type": "saccade",
    "second_delay": None
}

model = {
    "model_name": "targMin",
    "formula": "y ~  1 + ontarget*mss", 
    "second_formula": "y ~ 1 + saccade_amplitude",
    "tmin": -0.2,
    "tmax": 0.9,
    "use_splines": 5,
    "solver": "ridge",
    "alpha": "CV",
    "scoring": "rms",
    "second_delay": None ,
    "eeg_chns": 64
}

