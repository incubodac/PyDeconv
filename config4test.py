# config.py

events_of_interest = {
    "first_intercept_event_type": "1",
    "second_intercept_event_type": "0",
    "second_delay": None
}

model = {
    "model_name": "targMin",
    "formula": "y ~  1 + effect ", 
    "second_formula": "y ~ 1 ",
    "tmin": -0.2,
    "tmax": 0.6,
    "use_splines": None,
    "solver": "ridge",
    "scoring": "rms",
    "second_delay": None ,
    "eeg_chns": 1
}

