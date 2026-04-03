from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import threading
from scipy import signal


meas_samples= None
ref_samples = None

global repeats_remaining



def transmit(sdr, tx_meas_ch, buf_tx, num_samples, repeat, tx_start_event, rx_done_event):
    # Wait for rx to be ready if needed
    if tx_start_event is not None:
        # print("TX: Waiting for receiver to be ready...")
        tx_start_event.wait()
    
    # Enable transmit channels
    # print("Starting transmit!")
    tx_meas_ch.enable = True
    tx_ref_ch.enable = True
    
    # Transmit loop
    global repeats_remaining
    repeats_remaining = repeat
    while repeats_remaining > 0:
        sdr.sync_tx(buf_tx, num_samples)  # write to bladeRF
        repeats_remaining -= 1
        if rx_done_event is not None and rx_done_event.is_set():
            # print("TX: Receive signaled completion, stopping transmission early")
            break
        
    
    # Disable transmit channels
    # print("Stopping transmit")
    tx_meas_ch.enable = False
    
    # Signal that transmit is done
    if rx_done_event is not None:
        rx_done_event.wait()
    
    # print("Transmit completed")

def receive(sdr, rx_meas_ch, num_samples, tx_start_event, rx_done_event):
    global meas_samples, ref_samples, repeats_remaining, window
    
    # Create receive buffer
    bytes_per_sample = 4  # don't change this, it will always use int16s
    buf = bytearray(2 * 1024 * bytes_per_sample )  # for both measurement and reference
    
    # Create filter for recieved signal
    nyquist = sample_rate/2 
    low = (f_tone - 50e3) / nyquist
    high = (f_tone + 50e3) / nyquist
    b, a = signal.cheby2(4, 40, [low, high], 'bandpass')
    
    # Enable receive channels
    # print("Starting receive")
    rx_meas_ch.enable = True
    rx_ref_ch.enable = True

    # Signal to TX that we're ready to receive
    if tx_start_event is not None:
        tx_start_event.set()
    
    # Receive loop
    meas_buf = np.zeros(num_samples, dtype=np.complex64)  # storage for IQ samples
    ref_buf = np.zeros(num_samples, dtype=np.complex64)  # storage for IQ samples
    num_samples_read = 0
    condition = False
    while condition == False:
        if repeats_remaining < 30:
            condition = True
            # Continue receiving until we have enough samples or the transmit is done
            while num_samples_read < num_samples:
                if num_samples > 0:
                    num = min(len(buf) // (bytes_per_sample * 2), num_samples - num_samples_read)
                else:
                    num = len(buf) // bytes_per_sample
        
                sdr.sync_rx(buf, 2*num)  # Read into buffer
                samples = np.frombuffer(buf, dtype=np.int16)
                samples_scaled = samples / 1800.0  # Scale to -1 to 1 (its using 12 bit ADC)
                meas_samples_scaled = samples_scaled[0::4] + 1j * samples_scaled[1::4]
                ref_samples_scaled = samples_scaled[2::4] + 1j * samples_scaled[3::4]
                meas_buf[num_samples_read:num_samples_read+num] = meas_samples_scaled[0:num]  # Store buf in samples array
                ref_buf[num_samples_read:num_samples_read+num] = ref_samples_scaled[0:num]
                num_samples_read += num

    # Wait for transmit to finish before stopping receiver
    if rx_done_event is not None:
        # print("RX: Receive complete")
        rx_done_event.set()
    
    # Disable receive channels
    # print("Stopping receive")
    rx_meas_ch.enable = False
    rx_ref_ch.enable = False

    


    avg_pwr = np.mean(np.abs(meas_buf)**2)
    avg_pwr_db = 10*np.log10(avg_pwr)
    # print("Recieved signal average power in dB :", avg_pwr_db )


    meas_unfiltered = meas_buf[int(100e3):].copy()
    ref_unfiltered = ref_buf[int(100e3):].copy()
    meas_samples = signal.filtfilt(b, a, meas_unfiltered)
    ref_samples = signal.filtfilt(b, a, ref_unfiltered)



    # meas_samples = meas_buf[int(100e3):].copy()
    # ref_samples = ref_buf[int(100e3):].copy()
    # meas_samples = meas_samples - np.mean(meas_samples)
    # ref_samples = ref_samples - np.mean(ref_samples)


    


# Initialize device
sdr = _bladerf.BladeRF()
sdr.open

# Setup channels
rx_meas_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))
rx_ref_ch = sdr.Channel(_bladerf.CHANNEL_RX(1))

tx_meas_ch = sdr.Channel(_bladerf.CHANNEL_TX(0))
tx_ref_ch = sdr.Channel(_bladerf.CHANNEL_TX(1))

# Configuration parameters
sample_rate = 10e6
bandwidth = 500e3
start_freq = 1e9
num_of_steps = 51
freq_step = 34e6
rx_gain = 60  # -15 to 60 dB
tx_gain_meas = 0 # -15 to 60 dB
tx_gain_ref = 56 # -15 to 60 dB
num_samples = int(50e3) + int(100e3)
repeat = 50  # number of times to repeat our signal
repeats_remaining = repeat
epsilon_r = 1

print('Starting the Program')

# Configure RX channels

rx_meas_ch.sample_rate = sample_rate
rx_meas_ch.bandwidth = bandwidth
#rx_meas_ch.gain_mode = _bladerf.GainMode.Manual
#rx_meas_ch.gain = rx_gain
rx_meas_ch.bias_tee = True


rx_ref_ch.sample_rate = sample_rate
rx_ref_ch.bandwidth = bandwidth
#rx_ref_ch.gain_mode = _bladerf.GainMode.Manual
#rx_ref_ch.gain = rx_gain


# Configure TX channels

tx_meas_ch.sample_rate = sample_rate
tx_meas_ch.bandwidth = bandwidth
tx_meas_ch.gain = tx_gain_meas
tx_meas_ch.bias_tee = True


tx_ref_ch.sample_rate = sample_rate
tx_ref_ch.bandwidth = bandwidth
tx_ref_ch.gain = tx_gain_ref


# Print configuration
print("RX meas sample_rate:", sdr.get_sample_rate(_bladerf.CHANNEL_RX(0)))
print("RX meas bandwidth:", sdr.get_bandwidth(_bladerf.CHANNEL_RX(0)))
print("RX meas gain mode:", sdr.get_gain_mode(_bladerf.CHANNEL_RX(0)))
print("RX meas bias tee:", sdr.get_bias_tee(_bladerf.CHANNEL_RX(0)))

print("RX ref sample_rate:", sdr.get_sample_rate(_bladerf.CHANNEL_RX(1)))
print("RX ref bandwidth:", sdr.get_bandwidth(_bladerf.CHANNEL_RX(1)))
print("RX ref gain mode :", sdr.get_gain_mode(_bladerf.CHANNEL_RX(1)))

print("TX meas sample_rate:", sdr.get_sample_rate(_bladerf.CHANNEL_TX(0)))
print("TX meas bandwidth:", sdr.get_bandwidth(_bladerf.CHANNEL_TX(0)))
print("TX meas manual gain:", sdr.get_gain(_bladerf.CHANNEL_TX(0)))
print("TX meas bias tee:", sdr.get_bias_tee(_bladerf.CHANNEL_TX(0)))

print("TX ref sample_rate:", sdr.get_sample_rate(_bladerf.CHANNEL_TX(1)))
print("TX ref bandwidth:", sdr.get_bandwidth(_bladerf.CHANNEL_TX(1)))
print("TX ref manual gain:", sdr.get_gain(_bladerf.CHANNEL_TX(1)))

# Generate IQ samples to transmit (a simple tone)
t = np.arange(num_samples) / sample_rate
f_tone = 500e3
tone = np.exp(1j * 2 * np.pi * f_tone * t).astype(np.complex64)  # will be -1 to +1

# window = np.hanning(num_samples - int(100e3))
#samples_tx = samples_tx * window
tone_scaled = tone * 1800.0  # Scale to -1 to 1 (its using 12 bit DAC)

# Create properly interleaved buffer for MIMO
samples_tx = np.empty(4 * len(tone), dtype=np.int16)

# Interleave according to the MIMO pattern (Ch0 I/Q, Ch1 I/Q, Ch0 I/Q, Ch1 I/Q...)
# Channel 0, sample i
samples_tx[0::4]   = np.real(tone_scaled).astype(np.int16)  # I0[i]
samples_tx[1::4] = np.imag(tone_scaled).astype(np.int16)  # Q0[i]
samples_tx[2::4]   = np.real(tone_scaled).astype(np.int16)  # I1[i]
samples_tx[3::4] = np.imag(tone_scaled).astype(np.int16)  # Q1[i]    


buf_tx = samples_tx.tobytes()  # convert to bytes for transmit buffer



# Create events for thread synchronization
tx_start_event = threading.Event()
rx_done_event = threading.Event()  # Changed from rx_done_event


matrix_samples_sfcw = np.zeros((num_samples - int(100e3), num_of_steps), dtype=np.complex64)

signal_buffer1 = np.zeros( num_of_steps * (num_samples - int(100e3)), dtype=np.complex64)
signal_buffer2 = np.zeros( num_of_steps * (num_samples - int(100e3)), dtype=np.complex64)
signal_buffer3 = np.zeros( num_of_steps * (num_samples - int(100e3)), dtype=np.complex64)

for i in range(num_of_steps):

    tx_start_event.clear()
    rx_done_event.clear()

    current_freq = start_freq + i*freq_step

    rx_meas_ch.frequency = current_freq
    rx_ref_ch.frequency = current_freq
    tx_meas_ch.frequency = current_freq
    tx_ref_ch.frequency = current_freq

    

    # Setup synchronous streams
    sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X2, 
                    fmt=_bladerf.Format.SC16_Q11,
                    num_buffers=16,
                    buffer_size=16384,
                    num_transfers=8,
                    stream_timeout=3500)

    sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X2, 
                    fmt=_bladerf.Format.SC16_Q11,
                    num_buffers=16,
                    buffer_size=16384,
                    num_transfers=8,
                    stream_timeout=3500)

    # Create and start transmit thread
    tx_thread = threading.Thread(
        target=transmit,
        args=(sdr, tx_meas_ch, buf_tx, num_samples, repeat, tx_start_event, rx_done_event)
    )
    tx_thread.start()

    # Create and start receive thread
    rx_thread = threading.Thread(
        target=receive,
        args=(sdr, rx_meas_ch, num_samples, tx_start_event, rx_done_event)
    )
    rx_thread.start()

    # Wait for threads to complete
    tx_thread.join()
    rx_thread.join()

    conj = np.conjugate(tone[int(100e3):])

    


    signal_buffer1[i*(num_samples - int(100e3)): (i+1)*(num_samples - int(100e3))] = meas_samples
    signal_buffer2[i*(num_samples - int(100e3)): (i+1)*(num_samples - int(100e3))] = ref_samples



    compensated_signal = (meas_samples * conj) / (ref_samples * conj)

    signal_buffer3[i*(num_samples - int(100e3)): (i+1)*(num_samples - int(100e3))] = compensated_signal

    matrix_samples_sfcw[:, i] = compensated_signal

    





rx_meas_ch.bias_tee = False
tx_meas_ch.bias_tee = False

sdr.close

k_value = 128
window = np.hanning(num_of_steps)


averaged_data = np.mean(matrix_samples_sfcw, axis=0)
print("Size of averaged_data: ", np.shape(averaged_data))
averaged_data_windowed = averaged_data * window
range_profile = np.fft.ifft(averaged_data_windowed, k_value)

# Create time axis in nanoseconds
c = 3e8/np.sqrt(epsilon_r)  # Speed of light in medium

time_axis_ns = np.linspace(0, 1/(2 * freq_step), len(range_profile))

freq_axis = np.linspace(0, int(num_of_steps*freq_step), num_of_steps)/1e6

# Plot the result
plt.figure(figsize=(12, 12))

# Create time vector for x-axis of time domain plots
time_vector = np.arange(len(signal_buffer1)) / sample_rate






plt.subplot(4, 1, 1)
fft_size = 1024
num_segments = len(signal_buffer1) // fft_size
avg_spectrum = np.zeros(fft_size)
for i in range(num_segments):
    segment = signal_buffer1[i*fft_size:(i+1)*fft_size]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(segment)))**2
    avg_spectrum += spectrum
avg_spectrum /= num_segments
avg_spectrum_db = 10*np.log10(avg_spectrum)
freq_axis = np.linspace(start_freq, start_freq + (num_of_steps -1)* freq_step, fft_size) / 1e6
plt.plot(freq_axis, avg_spectrum_db)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Power [dB]")
plt.title("Measurement Channel Average Spectrum")
plt.grid(True)



# Reference channel average spectrum (similar structure)
plt.subplot(4, 1, 2)
fft_size = 1024
num_segments = len(signal_buffer2) // fft_size
avg_spectrum = np.zeros(fft_size)
for i in range(num_segments):
    segment = signal_buffer2[i*fft_size:(i+1)*fft_size]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(segment)))**2
    avg_spectrum += spectrum
avg_spectrum /= num_segments
avg_spectrum_db = 10*np.log10(avg_spectrum)
freq_axis = np.linspace(start_freq, start_freq + (num_of_steps -1)* freq_step, fft_size) / 1e6
plt.plot(freq_axis, avg_spectrum_db)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Power [dB]")
plt.title("Reference Channel Average Spectrum")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(time_vector * 1000, np.real(signal_buffer3), 'b-', label='I (Real)')
plt.plot(time_vector * 1000, np.imag(signal_buffer3), 'r-', label='Q (Imag)')
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.title("Compensated Signal Time Domain (I/Q)")
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time_axis_ns * 1e9, np.abs(range_profile))
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.title('Pulse Compression Result')
plt.grid(True)



peak_idx = np.argmax(np.abs(range_profile))

    
# Calculate the time at which the peak occurs (in seconds)
peak_time = time_axis_ns[peak_idx]
    
# Convert to nanoseconds for better readability
print('Peak was detected at time =', peak_time * 1e9, "nano seconds")



plt.tight_layout()
plt.show()