from scipy.signal import *


def get_best_order(freq):
    return int(freq / 10) + 1


def FREQUENCY_SCALING(x):  # use this so the freq you input is in 1/minutes aka "per minute"
    return (1 / 60) / x


def my_sosfilt(t, input_signal, cutoff_frequency, order, sampling_frequency=1/60):

    for f in cutoff_frequency:
        if FREQUENCY_SCALING(f) > FREQUENCY_SCALING(sampling_frequency / 2):
            print("cutoff frequency must be < fs/2, fs={}, fs/2={}".
                  format(FREQUENCY_SCALING(cutoff_frequency), FREQUENCY_SCALING(sampling_frequency / 2)))

    sos = butter(order, [FREQUENCY_SCALING(f) for f in cutoff_frequency], t, output="sos")

    y, _ = sosfilt(sos, input_signal, zi=sosfilt_zi(sos) * input_signal[0])

    return y


def lowpass(input_signal, cutoff_frequency, sampling_frequency=1 / 60):
    return my_sosfilt("lowpass", input_signal, [cutoff_frequency], get_best_order(cutoff_frequency), sampling_frequency)


def highpass(input_signal, cutoff_frequency, sampling_frequency=1 / 60):
    return my_sosfilt("highpass", input_signal, [cutoff_frequency], get_best_order(cutoff_frequency), sampling_frequency)


def bandpass(input_signal, lower_cutoff_frequency, upper_cutoff_frequency, sampling_frequency=1/60):
    return my_sosfilt("bandpass", input_signal,
                      [lower_cutoff_frequency, upper_cutoff_frequency],
                      get_best_order((upper_cutoff_frequency-lower_cutoff_frequency)/2), sampling_frequency)


def bandstop(input_signal, lower_cutoff_frequency, upper_cutoff_frequency, sampling_frequency=1/60):
    return my_sosfilt("bandstop", input_signal,
                      [lower_cutoff_frequency, upper_cutoff_frequency],
                      get_best_order((upper_cutoff_frequency-lower_cutoff_frequency)/2), sampling_frequency)
