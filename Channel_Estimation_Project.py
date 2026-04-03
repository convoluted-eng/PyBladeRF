from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import threading
import scipy
from scipy import signal

# Global variables
rx1_signal = None
rx2_signal = None

measurement_complete = False
H = None

global repeats_remaining


def transmit(sdr, tx_ch1, tx_ch2, buf_tx, num_samples, repeat, tx_start_event, rx_done_event):
    global measurement_complete, repeats_remaining
    
    if tx_start_event is not None:
        tx_start_event.wait()
    
    tx_ch1.enable = True
    tx_ch2.enable = True
    print("TX: Transmitting orthogonal pilots")
    
    repeats_remaining = repeat
    while repeats_remaining > 0 and not measurement_complete:
        sdr.sync_tx(buf_tx, num_samples)
        repeats_remaining -= 1
        if rx_done_event is not None and rx_done_event.is_set():
            break
    
    tx_ch1.enable = False
    tx_ch2.enable = False
    
    if rx_done_event is not None:
        rx_done_event.wait()
    
    print("TX: Complete")

def receive(sdr, rx_ch1, rx_ch2, tone, tone2, num_samples, tx_start_event, rx_done_event):
    global  measurement_complete, H, repeats_remaining, rx1_signal, rx2_signal
    
    bytes_per_sample = 4
    buf = bytearray(2 * 1024 * bytes_per_sample)

    
    
    rx_ch1.enable = True
    rx_ch2.enable = True
    print("RX: Ready")
    
    if tx_start_event is not None:
        tx_start_event.set()
    
    # Collect snapshots
    
    

    rx1_buf = np.zeros(num_samples, dtype=np.complex64)
    rx2_buf = np.zeros(num_samples, dtype=np.complex64)
    num_read = 0
        
    # Collect samples for this snapshot
    condition = False
    while condition == False:
        if repeats_remaining < 30:
            condition = True
            while num_read < num_samples:
                num = min(len(buf) // (bytes_per_sample * 2), num_samples - num_read)
                sdr.sync_rx(buf, 2 * num)
            
                samples = np.frombuffer(buf, dtype=np.int16)
                samples_scaled = samples / 1800.0
            
                rx1_scaled = samples_scaled[0::4] + 1j*samples_scaled[1::4]
                rx2_scaled = samples_scaled[2::4] + 1j*samples_scaled[3::4]
                rx1_buf[num_read:num_read+num] = rx1_scaled[0:num]  # Store buf in samples array
                rx2_buf[num_read:num_read+num] = rx2_scaled[0:num]
                num_read += num
        
    # Skip initial transient
    # rx1_unfiltered = rx1_buf.copy()
    # rx2_unfiltered = rx2_buf.copy()
    # rx1_signal = signal.filtfilt(b, a, rx1_unfiltered)
    # rx2_signal = signal.filtfilt(d, c, rx2_unfiltered)
    # rx1_signal = rx1_signal[int(10e3):]
    # rx2_signal = rx2_signal[int(10e3):]

    rx1_signal = rx1_buf[int(10e3):].copy()
    rx2_signal = rx2_buf[int(10e3):].copy()
        
    # Estimate channel matrix via cross-correlation
    H = np.zeros((2, 2), dtype=np.complex64)
    # pilot_length = len(rx1_signal) 
       
    for rx_idx, rx_signal in enumerate([rx1_signal, rx2_signal]):
        for tx_idx, pilot in enumerate([tone[int(10e3):], tone2[int(10e3):]]):
            # Cross-correlate received signal with pilot
            corr = scipy.signal.correlate(rx_signal, pilot, mode='valid')
            peak_idx = np.argmax(np.abs(corr))
            H[rx_idx, tx_idx] = corr[peak_idx] / np.linalg.norm(pilot)**2
            # H = (H - H.min()) / (H.max() - H.min())
        
    
    measurement_complete = True
    rx_ch1.enable = False
    rx_ch2.enable = False
    
    if rx_done_event is not None:
        rx_done_event.set()
    
    print("RX: Complete")
    

# Initialize device
sdr = _bladerf.BladeRF()

# Setup channels
rx_ch1 = sdr.Channel(_bladerf.CHANNEL_RX(0))
rx_ch2 = sdr.Channel(_bladerf.CHANNEL_RX(1))
tx_ch1 = sdr.Channel(_bladerf.CHANNEL_TX(0))
tx_ch2 = sdr.Channel(_bladerf.CHANNEL_TX(1))

# Configuration
sample_rate = 10e6
bandwidth = 500e3
center_freq = 2.4e9
tx_gain = 56
num_samples = int(50e3)
repeat = 50
repeats_remaining = repeat


# Configure all channels
for ch in [rx_ch1, rx_ch2, tx_ch1, tx_ch2]:
    ch.frequency = center_freq
    ch.sample_rate = sample_rate
    ch.bandwidth = bandwidth
    if ch in [tx_ch1, tx_ch2]:
        ch.gain = tx_gain

# Generate IQ samples to transmit (a simple tone)
t = np.arange(num_samples) / sample_rate
f_tone = 500e3
f_tone2 = 0.25*500e3

tone_unfiltered = np.exp(1j * 2 * np.pi * f_tone * t).astype(np.complex64)  # will be -1 to +1
tone_unfiltered2 = np.exp(1j * 2 * np.pi * f_tone2 * t).astype(np.complex64)  # will be -1 to +1

# Create filter for recieved signal
nyquist = sample_rate / 2
low = (f_tone - 50e3/2) / nyquist
high = (f_tone + 50e3/2) / nyquist
b, a = signal.cheby2(4, 40, [low, high], 'bandpass')

low2 = (f_tone2 - 50e3/2) / nyquist
high2 = (f_tone2 + 50e3/2) / nyquist
d, c = signal.cheby2(4, 40, [low2, high2], 'bandpass')

tone = signal.filtfilt(b, a, tone_unfiltered)
tone2 = signal.filtfilt(d, c, tone_unfiltered2)

tone_scaled = tone * 1800.0  # Scale to -1 to 1 (its using 12 bit DAC)
tone_scaled2 = tone2 * 1800.0 
# Create properly interleaved buffer for MIMO
samples_tx = np.empty(4 * len(tone), dtype=np.int16)

# Interleave according to the MIMO pattern (Ch0 I/Q, Ch1 I/Q, Ch0 I/Q, Ch1 I/Q...)
# Channel 0, sample i
samples_tx[0::4]  = np.real(tone_scaled).astype(np.int16)  # I0[i]
samples_tx[1::4] = np.imag(tone_scaled).astype(np.int16)  # Q0[i]
samples_tx[2::4]  = np.real(tone_scaled2).astype(np.int16)  # I1[i]
samples_tx[3::4] = np.imag(tone_scaled2).astype(np.int16)  # Q1[i]    


buf_tx = samples_tx.tobytes()  # convert to bytes for transmit buffer




# Thread synchronization
tx_start_event = threading.Event()
rx_done_event = threading.Event()


tx_start_event.clear()
rx_done_event.clear()

# Configure sync streams
sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X2, 
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=16, buffer_size=16384,
                num_transfers=8, stream_timeout=3500)

sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X2, 
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=16, buffer_size=16384,
                num_transfers=8, stream_timeout=3500)


# Start threads
tx_thread = threading.Thread(target=transmit, 
                            args=(sdr, tx_ch1, tx_ch2, buf_tx, num_samples, repeat, 
                                    tx_start_event, rx_done_event))
tx_thread.start()

rx_thread = threading.Thread(target=receive, 
                            args=(sdr, rx_ch1, rx_ch2, tone, tone2, num_samples, 
                                    tx_start_event, rx_done_event))

rx_thread.start()

# Wait for threads to complete
tx_thread.join()
rx_thread.join()
    



# Process results

print("\n===  Channel Matrix H ===")
print(H)

# SVD analysis
U, S, Vh = np.linalg.svd(H)
condition_number = S[0] / S[1] if S[1] > 1e-10 else np.inf
rank = np.sum(S > 1e-6)

print(f"\nSingular values: {S}")
print(f"Condition number: {condition_number:.2f}")
print(f"Rank: {rank}")

# Capacity estimation (assuming SNR = 20 dB)
SNR_linear = 100  # 20 dB
capacity = np.sum(np.log2(1 + SNR_linear * S**2))
print(f"MIMO capacity (20dB SNR): {capacity:.2f} bits/s/Hz")

# plt.figure
# plt.subplot(2, 2, 1)


# fft_result = np.fft.fftshift(np.fft.fft(tone))
# freqs = np.fft.fftshift(np.fft.fftfreq(len(tone), 1/sample_rate)) / 1e6
# power_db = 20 * np.log10(np.abs(fft_result) + 1e-12)
    
# plt.plot(freqs, power_db)
# plt.xlabel('Frequency [MHz]')
# plt.ylabel('Magnitude [dB]')
# plt.title(f'ZC Pilot 1 Spectrum')
# plt.grid(True)


# plt.subplot(2, 2, 2)

# fft_result = np.fft.fftshift(np.fft.fft(tone2))
# freqs = np.fft.fftshift(np.fft.fftfreq(len(tone2), 1/sample_rate)) / 1e6
# power_db = 20 * np.log10(np.abs(fft_result) + 1e-12)
    
# plt.plot(freqs, power_db)
# plt.xlabel('Frequency [MHz]')
# plt.ylabel('Magnitude [dB]')
# plt.title(f'ZC Pilot 2 Spectrum')
# plt.grid(True)


# plt.subplot(2, 2, 3)


# fft_result = np.fft.fftshift(np.fft.fft(rx1_signal))
# freqs = np.fft.fftshift(np.fft.fftfreq(len(rx1_signal), 1/sample_rate)) / 1e6
# power_db = 20 * np.log10(np.abs(fft_result) + 1e-12)
    
# plt.plot(freqs, power_db)
# plt.xlabel('Frequency [MHz]')
# plt.ylabel('Magnitude [dB]')
# plt.title(f'rx1_signal Spectrum')
# plt.grid(True)


# plt.subplot(2, 2, 4)

# fft_result = np.fft.fftshift(np.fft.fft(rx2_signal))
# freqs = np.fft.fftshift(np.fft.fftfreq(len(rx2_signal), 1/sample_rate)) / 1e6
# power_db = 20 * np.log10(np.abs(fft_result) + 1e-12)
    
# plt.plot(freqs, power_db)
# plt.xlabel('Frequency [MHz]')
# plt.ylabel('Magnitude [dB]')
# plt.title(f'rx2_signal Spectrum')
# plt.grid(True)


# plt.figure(figsize=(12, 12))

# # # channel 0 and 1 time domain - show both I and Q
# plt.subplot(4, 1, 1)
# plt.plot(t[int(10e3):], np.real(tone[int(10e3):]), 'b-', label='I (Real)')
# plt.plot(t[int(10e3):], np.imag(tone[int(10e3):]), 'r-', label='Q (Imag)')


# plt.ylabel("Amplitude")
# plt.title("Channel 0 Time Domain (I/Q)")
# plt.legend()
# plt.grid(True)

# plt.subplot(4, 1, 2)
# plt.plot(t[int(10e3):], np.real(tone2[int(10e3):]), 'b-', label='I (Real)')
# plt.plot(t[int(10e3):], np.imag(tone2[int(10e3):]), 'r-', label='Q (Imag)')


# plt.ylabel("Amplitude")
# plt.title("Channel 1 Time Domain (I/Q)")
# plt.legend()
# plt.grid(True)

# plt.subplot(4, 1, 3)
# plt.plot(t[int(10e3):], np.real(rx1_signal), 'b-', label='I (Real)')
# plt.plot(t[int(10e3):], np.imag(rx1_signal), 'r-', label='Q (Imag)')


# plt.ylabel("Amplitude")
# plt.title("rx1_signal Time Domain (I/Q)")
# plt.legend()
# plt.grid(True)

# plt.subplot(4, 1, 4)
# plt.plot(t[int(10e3):], np.real(rx2_signal), 'b-', label='I (Real)')
# plt.plot(t[int(10e3):], np.imag(rx2_signal), 'r-', label='Q (Imag)')


# plt.ylabel("Amplitude")
# plt.title("rx2_signal Time Domain (I/Q)")
# plt.legend()
# plt.grid(True)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# H matrix magnitude
im = axes[0, 0].imshow(np.abs(H), cmap='hot', interpolation='nearest')
axes[0, 0].set_title('|H| Matrix Magnitude')
axes[0, 0].set_xlabel('TX Antenna')
axes[0, 0].set_ylabel('RX Antenna')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_yticks([0, 1])
plt.colorbar(im, ax=axes[0, 0])

# H matrix phase
im2 = axes[0, 1].imshow(np.angle(H), cmap='hsv', interpolation='nearest')
axes[0, 1].set_title('∠H Matrix Phase [rad]')
axes[0, 1].set_xlabel('TX Antenna')
axes[0, 1].set_ylabel('RX Antenna')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
plt.colorbar(im2, ax=axes[0, 1])

# Singular values
axes[1, 0].bar([0, 1], S)
axes[1, 0].set_title('Singular Values (Spatial Modes)')
axes[1, 0].set_xlabel('Mode Index')
axes[1, 0].set_ylabel('Gain')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].grid(True, alpha=0.3)

# Capacity vs SNR
SNR_dB_range = np.arange(0, 31, 2)
SNR_linear_range = 10**(SNR_dB_range / 10)
capacity_curve = [np.sum(np.log2(1 + snr * S**2)) for snr in SNR_linear_range]
axes[1, 1].plot(SNR_dB_range, capacity_curve, 'b-', linewidth=2)
axes[1, 1].set_title('MIMO Capacity vs SNR')
axes[1, 1].set_xlabel('SNR [dB]')
axes[1, 1].set_ylabel('Capacity [bits/s/Hz]')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()