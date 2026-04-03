from bladerf import _bladerf
import numpy as np
import threading
from scipy import signal
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy import interpolate
import cv2 

# Global variables
meas_samples = None
ref_samples = None
global repeats_remaining

# Original functions kept intact
def transmit(sdr, tx_meas_ch, buf_tx, num_samples, repeat, tx_start_event, rx_done_event):
    # Wait for rx to be ready if needed
    if tx_start_event is not None:
        print("TX: Waiting for receiver to be ready...")
        tx_start_event.wait()
    
    # Enable transmit channels
    print("Starting transmit!")
    tx_meas_ch.enable = True
    
    # Transmit loop
    global repeats_remaining
    repeats_remaining = repeat
    while repeats_remaining > 0:
        sdr.sync_tx(buf_tx, num_samples)  # write to bladeRF
        repeats_remaining -= 1
        if rx_done_event is not None and rx_done_event.is_set():
            print("TX: Receive signaled completion, stopping transmission early")
            break
        
    
    # Disable transmit channels
    print("Stopping transmit")
    tx_meas_ch.enable = False
    
    # Signal that transmit is done
    if rx_done_event is not None:
        rx_done_event.wait()
    
    print("Transmit completed")

def receive(sdr, rx_meas_ch, num_samples, tx_start_event, rx_done_event):
    global meas_samples, ref_samples, repeats_remaining
    
    # Create receive buffer
    bytes_per_sample = 4  # don't change this, it will always use int16s
    buf = bytearray(2 * 1024 * bytes_per_sample )  # for both measurement and reference
    
    # Create filter for recieved signal
    nyquist = sample_rate/2 
    low = (f_tone - 50e3) / nyquist
    high = (f_tone + 50e3) / nyquist
    b, a = signal.cheby2(4, 40, [low, high], 'bandpass')
    
    # Enable receive channels
    print("Starting receive")
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
        print("RX: Receive complete")
        rx_done_event.set()
    
    # Disable receive channels
    print("Stopping receive")
    rx_meas_ch.enable = False
    rx_ref_ch.enable = False

    avg_pwr = np.mean(np.abs(meas_buf)**2)
    avg_pwr_db = 10*np.log10(avg_pwr)
    print("Recieved signal average power in dB :", avg_pwr_db )

    meas_unfiltered = meas_buf[int(100e3):].copy()
    ref_unfiltered = ref_buf[int(100e3):].copy()
    meas_samples = signal.filtfilt(b, a, meas_unfiltered)
    ref_samples = signal.filtfilt(b, a, ref_unfiltered)


# Main program
if __name__ == "__main__":
    # Initialize device
    sdr = _bladerf.BladeRF()

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
    num_samples = int(2048) + int(100e3)
    repeat = 50  # number of times to repeat our signal
    repeats_remaining = repeat
    epsilon_r = 1

    print('Starting the Program')

    # Configure RX channels
    rx_meas_ch.sample_rate = sample_rate
    rx_meas_ch.bandwidth = bandwidth
    rx_meas_ch.bias_tee = True

    rx_ref_ch.sample_rate = sample_rate
    rx_ref_ch.bandwidth = bandwidth

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

    print("RX ref sample_rate:", sdr.get_sample_rate(_bladerf.CHANNEL_RX(1)))
    print("RX ref bandwidth:", sdr.get_bandwidth(_bladerf.CHANNEL_RX(1)))
    print("RX ref gain mode :", sdr.get_gain_mode(_bladerf.CHANNEL_RX(1)))

    print("TX meas sample_rate:", sdr.get_sample_rate(_bladerf.CHANNEL_TX(0)))
    print("TX meas bandwidth:", sdr.get_bandwidth(_bladerf.CHANNEL_TX(0)))
    print("TX meas manual gain:", sdr.get_gain(_bladerf.CHANNEL_TX(0)))

    print("TX ref sample_rate:", sdr.get_sample_rate(_bladerf.CHANNEL_TX(1)))
    print("TX ref bandwidth:", sdr.get_bandwidth(_bladerf.CHANNEL_TX(1)))
    print("TX ref manual gain:", sdr.get_gain(_bladerf.CHANNEL_TX(1)))

    # Generate IQ samples to transmit (a simple tone)
    t = np.arange(num_samples) / sample_rate
    f_tone = 500e3
    tone = np.exp(1j * 2 * np.pi * f_tone * t).astype(np.complex64)  # will be -1 to +1
    tone *= 1800.0  # Scale to -1 to 1 (its using 12 bit DAC)

    # Create properly interleaved buffer for MIMO
    samples_tx = np.empty(2 * len(tone), dtype=np.int16)
    samples_tx[0::2] = np.real(tone).astype(np.int16)  # I0[i]
    samples_tx[1::2] = np.imag(tone).astype(np.int16)  # Q0[i]
    buf_tx = samples_tx.tobytes()  # convert to bytes for transmit buffer

    # Setup PyQtGraph
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(title="Real-time Radar Range Profile")
    win.resize(1000, 800)

    # Calculate expected max range
    c = 3e8/np.sqrt(epsilon_r)  # Speed of light in medium
    max_range = c / (2*freq_step)   # Maximum unambiguous range
    print(f"Maximum range: {max_range:.2f} meters")

    # Create standard grayscale color map for B-scan
    pos = np.array([0.0, 0.5, 1.0])  # Three points: min, mid, max
    color = np.array([
        [0, 0, 0, 255],         # Black (strong signal)
         [128, 128, 128, 255],   # Gray (background/zero)
        [255, 255, 255, 255]    # White (weak signal)
    ], dtype=np.ubyte)
    cmap = pg.ColorMap(pos, color)
    
    # A-scan plot
    p1 = win.addPlot(title="A-scan (Range Profile)")
    p1.setLabel('left', "Range", units='m')
    p1.setLabel('bottom', "Amplitude")
    p1.setYRange(0, max_range)
    p1.showGrid(x=True, y=True)
    p1.invertY(True)  # Invert y-axis so 0 range is at top
    
    # Create the range profile plot
    range_meters = np.linspace(0, max_range, 2048)  # Using standard IFFT size
    curve_a_scan = p1.plot(pen=pg.mkPen('b', width=2))
    curve_a_scan.setData(np.zeros(2048), range_meters)  # Initial empty plot
    
    win.nextRow()
    
    # B-scan plot
    p2 = win.addPlot(title="B-scan (Range Profile History)")
    p2.setLabel('left', "Range", units='m')
    p2.setLabel('bottom', "Position (scan number)")
    p2.invertY(True)
    p2.showGrid(x=True, y=True)
    
    # Create B-scan image
    b_scan_width = 10  # Width of the B-scan display
    interpolation_factor = 100
    b_scan_data = np.zeros((2048, b_scan_width), dtype=np.float32)
    b_scan_display = np.zeros((2048, interpolation_factor * b_scan_width), dtype=np.float32)
    # Create B-scan image item
    img_b_scan = pg.ImageItem()
    img_b_scan.setLookupTable(cmap.getLookupTable())
    p2.addItem(img_b_scan)
    
    # Set B-scan axis scales
    tr = QtGui.QTransform()
    tr.scale(1, max_range / 2048)
    img_b_scan.setTransform(tr)
    img_b_scan.setImage(b_scan_data.T)
    
    # Add status display
    win.nextRow()
    info_text = pg.LabelItem(justify='left')
    info_text.setText("Starting radar acquisition...")
    win.addItem(info_text)
    
    win.show()

    latest_range_profile = None
    latest_peak_distance = 0
    display_queue = []  # Queue to hold new profiles if acquisition is faster than display

    # Set up QTimer for display updates
    def update_display():
        global b_scan_data, abs_profile, peak_distance, b_scan_counter, square_buffer
        # Update A-scan
        curve_a_scan.setData(abs_profile / np.max(abs_profile), range_meters)
    
        # Update B-scan
        b_scan_data = np.roll(b_scan_data, -1, axis=1)
        b_scan_data[:, -1] = (abs_profile - np.min(abs_profile)) / (np.max(abs_profile) - np.min(abs_profile))

        x_original = np.arange(b_scan_width)
        x_new = np.linspace(0, b_scan_width-1, b_scan_width * interpolation_factor)
    
        for row in range(2048):
            f = interpolate.interp1d(x_original, b_scan_data[row, :], kind='linear')
            b_scan_display[row, :] = f(x_new)
    
        # Update B-scan image
        img_b_scan.setImage(
            b_scan_display.T,
            autoLevels=False,
            levels=(-np.max(np.abs(b_scan_data))/2, np.max(np.abs(b_scan_data)))
        )

        # Update info text
        info_text.setText(f"Range profile #{scan_count} ")
    
        # Process events to ensure UI updates immediately
        app.processEvents()
    
    def b_scan_image_processing():
        global b_scan_data,image_index, max_range
        normalized_data = (b_scan_display - np.min(b_scan_data)) / (np.max(b_scan_data) - np.min(b_scan_data))
        b_scan_uint8 = (normalized_data * 255).astype(np.uint8)

        otsu_thresh, otsu_binary = cv2.threshold(b_scan_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        adaptive_mean = cv2.adaptiveThreshold(b_scan_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
        adaptive_gaussian = cv2.adaptiveThreshold(b_scan_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)

        height = otsu_binary.shape[0]

        for range_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            pixel_pos = int((range_val / max_range) * height)
            text = f'{range_val:.1f}m'
    
            cv2.putText(otsu_binary, text, (5, pixel_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(adaptive_mean, text, (5, pixel_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(adaptive_gaussian, text, (5, pixel_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


        # Save the image
        cv2.imwrite(f'otsu_thresholded_{image_index}.jpeg', otsu_binary)
        cv2.imwrite(f'adaptive_mean_{image_index}.jpeg', adaptive_mean)
        cv2.imwrite(f'adaptive_gaussian_{image_index}.jpeg', adaptive_gaussian)

        # Display the image
        cv2.imshow('OTSU Thresholded B-scan', otsu_binary)
        cv2.imshow('Adaptive Mean Thresholded B-scan', adaptive_mean)
        cv2.imshow('Adaptive Gaussian Thresholded B-scan', adaptive_gaussian)
        cv2.waitKey(1) 

        image_index += 1

        

    
    # Main acquisition loop
    is_running = True
    scan_count = 0
    b_scan_counter = b_scan_width
    square_buffer = np.zeros((b_scan_width, 2, 5))
    image_index = 1

    try:
        while is_running:
            # Create events for thread synchronization
            tx_start_event = threading.Event()
            rx_done_event = threading.Event()
            
            # Initialize matrix for frequency sweep data
            matrix_samples_sfcw = np.zeros((num_samples - int(100e3), num_of_steps), dtype=np.complex64)
            
            info_text.setText(f"Acquiring range profile #{scan_count+1}...")
            
            # Frequency sweep
            for i in range(num_of_steps):
                tx_start_event.clear()
                rx_done_event.clear()
                
                app.processEvents()

                # Set frequency for this step
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
                
                # Transmit and receive
                tx_thread = threading.Thread(
                    target=transmit,
                    args=(sdr, tx_meas_ch, buf_tx, num_samples, repeat, tx_start_event, rx_done_event)
                )
                tx_thread.start()
                
                rx_thread = threading.Thread(
                    target=receive,
                    args=(sdr, rx_meas_ch, num_samples, tx_start_event, rx_done_event)
                )
                rx_thread.start()
                
                # Wait for threads to complete
                tx_thread.join()
                rx_thread.join()
                
                # Process data for this frequency step

                compensated_signal = meas_samples / (ref_samples)
                matrix_samples_sfcw[:, i] = compensated_signal

                print(f"Frequency steps remaining: {num_of_steps - i}")

                # Check if window was closed
                if not win.isVisible():
                    is_running = False
                    break
            
            if not is_running:
                break
            
            # Process the complete frequency sweep data
            averaged_data = np.mean(matrix_samples_sfcw, axis=0)
            
            
            # Calculate range profile
            range_profile = np.fft.ifft(averaged_data, 2048)
            
            # Find peak for display
            peak_idx = np.argmax(np.abs(range_profile))
            peak_time = np.linspace(0, 1/freq_step, len(range_profile))[peak_idx]
            peak_distance = c * peak_time/2
            abs_profile = np.abs(range_profile)

            peak_value = abs_profile[peak_idx]

            #threshold = 0.20 * peak_value
            #thresholded_profile = np.zeros_like(abs_profile)
            #thresholded_profile[abs_profile > threshold] = abs_profile[abs_profile > threshold]
            
            
            # Update visualization
            update_display()
            
            #update b_scan index counter
            if b_scan_counter > 0:
                b_scan_counter -= 1
            else:
                #Call here the image processing.
                b_scan_image_processing()
                b_scan_counter = b_scan_width

            # Update info text
            info_text.setText(f"Range profile #{scan_count+1}: Peak at {peak_distance:.2f} meters")
            print(f"Range profile #{scan_count+1}: Peak at {peak_distance:.2f} meters")
            scan_count += 1
            
            # Process events
            app.processEvents()
            
            # Check if window was closed
            if not win.isVisible():
                is_running = False
    
    except KeyboardInterrupt:
        print("\nAcquisition stopped by user")
    except Exception as e:
        print(f"\nError during acquisition: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        rx_meas_ch.bias_tee = False
        tx_meas_ch.bias_tee = False
