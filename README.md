# Classical Laser Emulation of Photon Entanglement: Quantum State Tomography and Bell Inequality Test
A Python analysis code for processing pulsed laser video recording for the experiment of "Quantum Tomography And Bell Test".
In the experiment, a series of laser pulses with different polarization were recorded via two detection arms (Alice and Bob), each equipped 
with a polarizing beam splitter and camera-based detectors. The analysis code reads the recorded videos, find coincident events, 
reconstructs the density matrix of the emulated Bell state, and tests the CHSH Bell inequality. 

Quantum Tomography: 
1. reconstruction and plotting of density matrices via quantum state tomography from the measured data
2. reconstruction and plotting of density matrices via quantum state tomography from the simulated data, providing a check of the analysis code.

Bell Test: calculation of the the S parameter and its uncertainty using coincidence measurements and Poisson statistics.

## File Description
### File: QT_and_Bell_Test.py
* __Inputs__: 
   * Part A: path to the directory specified by 'PART_A_DIR' with MP4 video files named '10bits.mp4', '25bits.mp4', '50bits.mp4' 
   * Part B: path to the directory specified by 'PART_B_DIR' MP4 video files named '{alpha}to{beta}.mp4' (for example: '0to22.5.mp4')
     * _Note_: Part B video files must follow the naming convention `{alpha}to{beta}.mp4` 
     where alpha and beta are the polarizer angles in degrees (e.g. `-45to22.5.mp4`). 
     Negative angles and decimal values are supported.
*__Control Parameters__:
   * selection of the desired analysis stage (Part A, Part B, Simulation)
   * selection of signal and video analysis parameters (shown below in 'How to Run' section)
*__Analysis__:
   * the code reads MP4 video files, 
   * extracts per-frame red-channel intensity inside user-defined regions of interest, 
   * detects laser pulse events by detecting signals that surpass a user-defined threshold
* __Quantum Tomography__:
    * Counts coincidences between Alice and Bob and calculates normalized intensity
  Reconstructs and plots the two Bell's states' density matrices, both for measured data and simulated pulses.
* __Bell Test__:
    * computes the CHSH correlation parameter S with its statistical uncertainty.
* __Outputs__:
   * _'PartA_QT_density.png'_ - 3D plot bar charts of the 
     reconstructed density matrices |ρᵢⱼ| for both |Φ⁺⟩ and |Ψ⁺⟩ states, 
     at each pulse count (10, 25, 50 bits). Corresponds to Fig. (3) in the paper.
   * _'PartB_QT_grid.png'_ - 4x4 grid of normalized Alice/Bob intensity 
     traces for all 16 angle combinations, with the  specific pair of angles as the title of each subplot. Corresponds to Fig. (5) in the paper.
   * _'PartA_simulation_verification_50.png'_ - a plot comparing the ideal and reconstructed density matrices from the simulation corresponds to Fig. (4) in the paper.
   - _Console output prints_ including:
     - the two Bell's states density matrices, both for measured data and simulated pulses.
     - E(α,β) values.
     - S ± error.
     - per-video coincidence counts.
## How to Run
### Step 1: define the paths
__define the paths for the directories with the data__
```python
PART_A_DIR = "path to your parta videos"
PART_B_DIR = "path to your partb videos"
```
### Step 2: Which Analysis To run
__define the parts of the analysis you'd like to run__
```python
DO_PART_A        = True   # reconstruct density matrices from Part A videos
DO_PART_B        = True   # compute CHSH parameter from Part B videos
RUN_SIMULATION_A = True   # run simulation for pipeline validation (no videos needed)
SHOW_PLOTS       = True   # display figures interactively. All figures are saved as PNG files in the working directory regardless 
                          # of whether 'SHOW_PLOTS' is True or False.
```
### Step 3: Analysis Parameters
__define the signal and video analysis parameters.__ These parameters may need adjustment for different 
experimental conditions

| Parameter | Default | Effect |
|-------------------|------|----------------------------------------------------------------------------------|
| 'EVENT_THRESHOLD' | 0.89 | Minimum normalized intensity to count as a detected signal (and not background). |
| 'MIN_EVENT_SEPARATION' | 30 | Minimum frame gap between two distinct signals. |
| 'COINCIDENCE_WINDOW' | 12 | Maximum frame difference for two events to be counted as coincident. |
| 'RED_START_THRESHOLD' | 2000 | Minimum raw red-pixel sum to mark the start of usable data in a video. |

__define the simulation parameters.__

| Parameter | Default | Effect |
|--------------------|------|------------------------------------------|
| 'RUN_SIMULATION_A' | True | True = run signal simulation for Part A. |
| 'SIM_A_SHOTS' | 50 |number of pulses to simulate. |
| 'SIM_A_SEED' | 1 | random seed for reproducibility. |

### Step 4: Run
On the first run, an interactive window will open for drawing 
rectangles around each detector's laser spot on the camera frame. 
Draw according to the part of the analysis you're working on (4 rectangles for Part A and 2 for Part B). 
These coordinates are saved automatically for each part and can be reused for future runs.
