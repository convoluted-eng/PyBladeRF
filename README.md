# PyBladeRF

Python implementations for BladeRF SDR transmit/receive testing. Provides SISO and MIMO configurations for RF system development and validation.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

Two Python scripts for BladeRF SDR testing:

- **SISO**: Single-channel TX/RX implementation
- **MIMO**: Dual-channel TX/RX implementation

Both include real-time signal processing, synchronized operations, and comprehensive analysis tools.

## Requirements

### Hardware
- BladeRF SDR device (2.0 micro A4 or compatible)
- USB 3.0 connection
- SMA cables or antennas
- RF attenuators (recommended for loopback testing)

### Software
- Python 3.7+
- BladeRF drivers and libraries
- Required packages: `numpy`, `matplotlib`, `scipy`, `bladerf`

## Installation

1. Install BladeRF drivers:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install bladerf libbladerf-dev
   ```

2. Install Python dependencies:
   ```bash
   pip install numpy matplotlib scipy bladerf
   ```

3. Clone repository:
   ```bash
   git clone https://github.com/convoluted-eng/PyBladeRF.git
   cd PyBladeRF
   ```

## Usage

### SISO Implementation
```bash
python siso_tx_rx.py
```
- Frequency: 2.4 GHz
- Test signal: 1 MHz tone
- Sample rate: 10 MSPS

### MIMO Implementation
```bash
python mimo_tx_rx.py
```
- Frequency: 1 GHz
- Test signal: 500 kHz tone
- Sample rate: 10 MSPS
- Channels: 2x2 MIMO

## Configuration

Key parameters can be modified in each script:

| Parameter | SISO | MIMO | Description |
|-----------|------|------|-------------|
| `center_freq` | 2.4 GHz | 1 GHz | RF center frequency |
| `f_tone` | 1 MHz | 500 kHz | Test tone frequency |
| `tx_gain` | -10 dB | 56 dB | Transmit power |
| `num_samples` | 1M | 20k | Samples per transmission |

## Test Setup

### Loopback Testing
```
BladeRF TX → [30dB Attenuator] → BladeRF RX
```

### Over-the-Air Testing
```
BladeRF TX → [Antenna] → [Antenna] → BladeRF RX
```

## Output

Both scripts generate:
- Frequency domain spectrum analysis
- Time domain I/Q plots
- Signal power measurements
- Transmitted vs received signal comparison

## Applications

- SDR hardware validation
- RF system prototyping
- Communication link testing
- Educational demonstrations
- Research platform for MIMO systems

## Safety

- Use RF attenuators in loopback configurations
- Verify local RF transmission regulations
- Start with low transmit power

## Troubleshooting

**Device not found**: Check USB connection and drivers
**Permission errors**: Add user to `plugdev` group (Linux)
**Sample drops**: Ensure USB 3.0 connection

## License

MIT License - see LICENSE file for details.

## Contributing

Pull requests welcome. For major changes, please open an issue first.

## Resources

- [BladeRF Documentation](https://github.com/Nuand/bladeRF/wiki)
- [Nuand BladeRF](https://www.nuand.com/)
- [PySDR Tutorial](https://pysdr.org/)
