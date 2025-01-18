# NV_Monitor

What is NV_Monitor?
------------------

NV_Monitor is a local web-based monitoring tool for NVIDIA GPUs. It provides real-time monitoring of multiple NVIDIA GPUs on your system through an intuitive web interface, making it easy to track GPU performance and utilization from any browser.

![NV_Monitor interface](/screenshot/NV_Monitor.png)

Table of Contents
----------------

- [NV\_Monitor](#nv_monitor)
  - [What is NV\_Monitor?](#what-is-nv_monitor)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Dependencies Installation](#dependencies-installation)
  - [Usage](#usage)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)

Features
--------

- Real-time monitoring of multiple NVIDIA GPUs
- Web-based interface accessible from any browser
- Monitors key metrics including:
  - GPU utilization
  - Memory usage
  - Temperature
  - Power consumption
  - Fan speed
  - Running processes
- Responsive design that works on desktop and mobile devices
- No external dependencies beyond NVIDIA drivers

Requirements
-----------

- NVIDIA GPU(s)
- NVIDIA drivers installed
- Modern web browser
- Python 3.6 or higher
- NVIDIA Management Library (NVML)

Dependencies Installation
-----------
In an python environment, run the following command to install the environment.

```bash
pip install -r requirements.txt
```

Usage
-----

1. Start the NV_Monitor server:

In the python environment, run the following command to start the server.

```bash
python main.py
```

2. Access the NV_Monitor interface:

Open your web browser and navigate to `http://localhost:8080`.


Troubleshooting
--------------

- If you can't connect to the web interface:
  - Verify the server is running
  - Check if the port 8080 is not being used by another application
  - Ensure your firewall allows connections to the port

- If GPU data is not showing:
  - Verify NVIDIA drivers are properly installed
  - Check if nvidia-smi works from command line
  - Ensure you have proper permissions to access GPU information

License
-------

NV_Monitor is licensed under the MIT License. See the LICENSE file for details.