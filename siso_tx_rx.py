"""
SISO TX/RX Implementation for BladeRF SDR
=========================================

This script implements a SISO (Single Input Single Output) transmit and receive
system using the BladeRF Software Defined Radio platform. It demonstrates:

- Single-channel transmission and reception
- Synchronized TX/RX operations using threading
- Real-time signal processing with filtering
- Spectrum analysis and visualization

Features:
- Transmits a 1 MHz tone at 2.4 GHz center frequency
- Concurrent transmission and reception with precise timing
- Bandpass filtering for received signals
- Comprehensive signal analysis and plotting

Hardware Requirements:
- BladeRF SDR device
- Appropriate RF connections (loopback or antenna setup)

Author: convoluted-eng
Date: 2025.07
License: MIT Liscense
"""

from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import threading
from scipy import signal

# =============================================================================
# Global Variables
# =============================================================================

meas_samples = None
repeats_remaining = None


# =============================================================================
# Transmit Function
# =============================================================================

def transmit(sdr, tx_meas_ch, buf_tx, num_samples, repeat, tx_start_event, rx_done_event):
    """
    Handle the transmission of IQ samples on the measurement channel.
    
    Args:
        sdr: BladeRF SDR object
        tx_meas_ch: Transmit measurement channel
        buf_tx: Buffer containing IQ samples to transmit
        num_samples: Number of samples per transmission
        repeat: Number of times to repeat the transmission
        tx_start_event: Threading event to synchronize with receiver
        rx_done_event: Threading event to signal completion
    """
    global repeats_remaining
    
    # Wait for receiver to be ready
    if tx_start_event is not None:
        print("TX: Waiting for receiver to be ready...")
        tx_start_event.wait()
    
    # Enable transmit channel
    print("Starting transmit!")
    tx_meas_ch.enable = True
    
    # Transmission loop
    repeats_remaining = repeat
    while repeats_remaining > 0:
        sdr.sync_tx(buf_tx, num_samples)  # Transmit to BladeRF
        print(f"Transmitting, repeats remaining: {repeats_remaining}")
        repeats_remaining -= 1
        
        # Check if receiver has signaled completion
        if rx_done_event is not None and rx_done_event.is_set():
            print("TX: Receive signaled completion, stopping transmission early")
            break
    
    # Disable transmit channel
    print("Stopping transmit")
    tx_meas_ch.enable = False
    
    # Wait for receiver to complete
    if rx_done_event is not None:
        rx_done_event.wait()
    
    print("Transmit completed")


# =============================================================================
# Receive Function
# =============================================================================

def receive(sdr, rx_meas_ch, num_samples, tx_start_event, rx_done_event):
    """
    Handle the reception and processing of IQ samples on the measurement channel.
    
    Args:
        sdr: BladeRF SDR object
        rx_meas_ch: Receive measurement channel
        num_samples: Number of samples to collect
        tx_start_event: Threading event to synchronize with transmitter
        rx_done_event: Threading event to signal completion
    """
    global meas_samples, repeats_remaining
    
    # Create receive buffer (4 bytes per sample for I/Q int16 data)
    bytes_per_sample = 4
    buf = bytearray(2 * 1024 * bytes_per_sample)
    
    # Design bandpass filter for received signal (centered at 1 MHz)
    nyquist = sample_rate / 2
    low = (1e6 - 100e3/2) / nyquist
    high = (1e6 + 100e3/2) / nyquist
    b, a = signal.butter(4, [low, high], 'bandpass')
    
    # Enable receive channel
    print("Starting receive")
    rx_meas_ch.enable = True
    
    # Signal to transmitter that receiver is ready
    if tx_start_event is not None:
        tx_start_event.set()
    
    # Initialize receive buffer
    x = np.zeros(num_samples, dtype=np.complex64)
    num_samples_read = 0
    condition = False
    
    # Reception loop - wait for sufficient transmission cycles
    while condition == False:
        if repeats_remaining < 30:
            condition = True
            
            # Continue receiving until we have enough samples
            while num_samples_read < num_samples:
                if num_samples > 0:
                    num = min(len(buf) // (bytes_per_sample * 2), num_samples - num_samples_read)
                else:
                    num = len(buf) // bytes_per_sample
        
                # Read IQ data from single channel
                sdr.sync_rx(buf, num)
                samples = np.frombuffer(buf, dtype=np.int16)
                
                # Scale from 12-bit ADC range to normalized values
                samples_scaled = samples / 1800.0
                
                # Convert interleaved I/Q samples to complex format
                samples_scaled = samples_scaled[0::2] + 1j * samples_scaled[1::2]
                
                # Store samples in buffer
                x[num_samples_read:num_samples_read+num] = samples_scaled[0:num]
                num_samples_read += num

    # Signal completion to transmitter
    if rx_done_event is not None:
        print("RX: Receive complete")
        rx_done_event.set()
    
    # Disable receive channel
    print("Stopping receive")
    rx_meas_ch.enable = False

    # Display reception statistics
    print("Is x complex =", np.iscomplexobj(x))
    print("First 20 values of x =", x[0:20])

    avg_pwr = np.mean(np.abs(x)**2)
    avg_pwr_db = 10*np.log10(avg_pwr)
    print("Received signal average power in dB:", avg_pwr_db)

    # Apply filtering to remove initial transients and filter signal
    meas_unfiltered = x[int(18.5e3):].copy()
    meas_samples = signal.filtfilt(b, a, meas_unfiltered)

    # Alternative: Use unfiltered samples (currently commented out)
    # meas_samples = x.copy()


# =============================================================================
# SDR Configuration
# =============================================================================

# Initialize BladeRF device
sdr = _bladerf.BladeRF()

# Setup channel objects
rx_meas_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))  # RX measurement channel
tx_meas_ch = sdr.Channel(_bladerf.CHANNEL_TX(0))  # TX measurement channel

# Configuration parameters
sample_rate = 10e6       # 10 MSPS
bandwidth = 500e3        # 500 kHz bandwidth
center_freq = 2.4e9      # 2.4 GHz center frequency
rx_gain = 0              # RX gain: -15 to 60 dB
tx_gain = -10            # TX gain: -15 to 60 dB
num_samples = int(1e6)   # 1M samples per transmission
repeat = 50              # Number of transmission cycles
repeats_remaining = repeat

# Calculate and display transmission duration
transmission_duration = num_samples / sample_rate * repeat
print(f'Duration of transmission: {transmission_duration} seconds')

# Configure RX measurement channel
rx_meas_ch.frequency = center_freq
rx_meas_ch.sample_rate = sample_rate
rx_meas_ch.bandwidth = bandwidth
# Note: Gain settings are commented out for automatic gain control
# rx_meas_ch.gain_mode = _bladerf.GainMode.Manual
# rx_meas_ch.gain = rx_gain

# Configure TX measurement channel
tx_meas_ch.frequency = center_freq
tx_meas_ch.sample_rate = sample_rate
tx_meas_ch.bandwidth = bandwidth
tx_meas_ch.gain = tx_gain


# =============================================================================
# Display Configuration
# =============================================================================

print("=== RX Configuration ===")
print("RX meas sample_rate:", sdr.get_sample_rate(_bladerf.CHANNEL_RX(0)))
print("RX meas bandwidth:", sdr.get_bandwidth(_bladerf.CHANNEL_RX(0)))
print("RX meas frequency:", sdr.get_frequency(_bladerf.CHANNEL_RX(0)))
print("RX meas gain mode:", sdr.get_gain_mode(_bladerf.CHANNEL_RX(0)))

print("\n=== TX Configuration ===")
print("TX meas sample_rate:", sdr.get_sample_rate(_bladerf.CHANNEL_TX(0)))
print("TX meas bandwidth:", sdr.get_bandwidth(_bladerf.CHANNEL_TX(0)))
print("TX meas frequency:", sdr.get_frequency(_bladerf.CHANNEL_TX(0)))
print("TX meas manual gain:", sdr.get_gain(_bladerf.CHANNEL_TX(0)))


# =============================================================================
# Signal Generation
# =============================================================================

# Generate time vector and test tone (1 MHz complex exponential)
t = np.arange(num_samples) / sample_rate
f_tone = 1e6
tone = np.exp(1j * 2 * np.pi * f_tone * t).astype(np.complex64)

# Optional windowing (currently disabled)
# window = np.hanning(num_samples)
# tone = tone * window

# Scale for 12-bit DAC range
tone_scaled = tone * 1800.0

# Create properly interleaved I/Q samples for single channel
samples_tx = np.empty(2 * len(tone), dtype=np.int16)
samples_tx[0::2] = np.real(tone_scaled).astype(np.int16)  # I samples (even indices)
samples_tx[1::2] = np.imag(tone_scaled).astype(np.int16)  # Q samples (odd indices)

# Convert to byte buffer for transmission
buf_tx = samples_tx.tobytes()


# =============================================================================
# Stream Configuration
# =============================================================================

# Configure synchronous TX stream (single channel)
sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1, 
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=16,
                buffer_size=16384,
                num_transfers=8,
                stream_timeout=3500)

# Configure synchronous RX stream (single channel)
sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1, 
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=16,
                buffer_size=16384,
                num_transfers=8,
                stream_timeout=3500)


# =============================================================================
# Thread Execution
# =============================================================================

# Create synchronization events
tx_start_event = threading.Event()
rx_done_event = threading.Event()

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

# Wait for both threads to complete
tx_thread.join()
rx_thread.join()


# =============================================================================
# Results and Visualization
# =============================================================================

print("Length of measurement samples signal =", len(meas_samples))

# Create comprehensive visualization
plt.figure(figsize=(12, 12))

# Analysis parameters for time domain plots
start_time_ms = 30      # Start time in milliseconds
end_time_ms = 30.04     # End time in milliseconds

# 1. Measurement channel average spectrum
plt.subplot(3, 1, 1)
fft_size = 2048
num_segments = len(meas_samples) // fft_size
avg_spectrum = np.zeros(fft_size)

for i in range(num_segments):
    segment = meas_samples[i*fft_size:(i+1)*fft_size]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(segment)))**2
    avg_spectrum += spectrum

avg_spectrum /= num_segments
avg_spectrum_db = 10*np.log10(avg_spectrum)
freq_axis = np.linspace(center_freq - sample_rate/2, center_freq + sample_rate/2, fft_size) / 1e6

plt.plot(freq_axis, avg_spectrum_db)
plt.ylabel("Power [dB]")
plt.title("Measurement Channel Average Spectrum")
plt.grid(True)

# 2. Measurement channel time domain (I/Q components)
plt.subplot(3, 1, 2)
plt.plot(t[int(18.5e3):] * 1000, np.real(meas_samples), 'b-', label='I (Real)')
plt.plot(t[int(18.5e3):] * 1000, np.imag(meas_samples), 'r-', label='Q (Imag)')
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.title("Measurement Channel Time Domain (I/Q)")
plt.legend()
plt.grid(True)

# 3. Original transmitted tone for comparison
plt.subplot(3, 1, 3)
plt.plot(t[int(18.5e3):] * 1000, np.real(tone[int(18.5e3):]), 'b-', label='I (Real)')
plt.plot(t[int(18.5e3):] * 1000, np.imag(tone[int(18.5e3):]), 'r-', label='Q (Imag)')
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.title("Original Transmitted Tone (I/Q)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
