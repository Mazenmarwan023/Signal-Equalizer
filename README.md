# Signal Equalizer Application

## Introduction

The Signal Equalizer is a versatile desktop application designed for modifying the magnitude of specific frequency components in audio and other signal types. This tool is widely used in music, speech, and biomedical applications, such as hearing aids and ECG analysis.

## Features

1. **Multiple Modes of Operation**:
   - **Uniform Range Mode**: Divides the total frequency range into 10 equal segments, each controlled by a slider.
   ![uniform mode]()

   - **Music and animals Mode**: Allows control of the magnitude of specific musical instruments and animals sounds in a mixed signal.
   ![music and animals mode]()

   - **Music and vowels mode**: Enables control of the magnitude of specific music instruments sounds and specific vowels from real life song.
   
   - **Wiener filter mode**: Enables the reduction of noise and enhancement of signal quality by optimally estimating the original signal from noisy observations.
    ![wiener mode]() 
   

2. **Fourier Transform Visualization**:
   - Displays the Fourier transform of the input signal.
   - Allows switching between **linear scale** and **Audiogram scale** for frequency visualization.

3. **Linked Cine Signal Viewers**:
   - Synchronized viewers for input and output signals.
   - Full functionality, including play, stop, pause, speed control, zoom, pan, and reset.

4. **Spectrograms**:
   - Visual representation of the input and output signals' frequency content.
   - Automatically updates to reflect slider changes.
   - Option to toggle spectrogram visibility.

5. **Intuitive UI**:
   - Seamless switching between modes via menus or dropdowns.
   - Dynamically updated slider labels and functionality based on the selected mode.
     
   *Screenshot for UI*
   ![UI]()


## Usage

1. **Load a Signal**:
   - Open a WAV or other supported file format.

2. **Choose a Mode**:
   - Select one of the four available modes:
     - Uniform Range Mode
     - Music and Animals Mode

     - Music and vowels Mode
     - Wiener Filter Mode

3. **Adjust Sliders**:
   - Modify frequency components using sliders.
   - Observe real-time updates in the output spectrogram and signal viewer.

4. **Save Output**:
   - Save the modified signal as a new file.

## Notes

- Ensure proper slider-to-frequency mapping in non-uniform modes.
- The Fourier transform visualization is crucial for validating frequency manipulations.
- Synchronized cine viewers ensure an accurate time-domain representation.

#### **Demo**


## **Setup**

- Clone the repo
```bash
git clone https://github.com/mahmoudmohamed22/Signal-Equalizer.git
```
- Enter Project Folder
```bash
cd Signal-Equalizer
```
- Install the requirements
```bash
pip install -r requirements.txt
```
- Run the Application
```bash
python main.py
```

## Contributors <a name = "Contributors"></a>
<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Mazenmarwan023" target="_black">
    <img src="https://avatars.githubusercontent.com/u/127551364?v=4" width="150px;" alt="Mazen Marwan"/>
    <br />
    <sub><b>Mazen Marwan</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/seiftaha" target="_black">
    <img src="https://avatars.githubusercontent.com/u/127027353?v=4" width="150px;" alt="Seif Taha"/>
    <br />
    <sub><b>Seif Taha</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/mahmoudmo22" target="_black">
    <img src="https://avatars.githubusercontent.com/u/56477186?v=4" width="150px;" alt="Mahmoud Mohamed"/>
    <br />
    <sub><b>Mahmoud Mohamed</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/farha1010" target="_black">
    <img src="https://avatars.githubusercontent.com/u/111386862?v=4" width="150px;" alt="Farha El-sayed "/>
    <br />
    <sub><b>Farha El-sayed </b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/Alyaaa16" target="_black">
    <img src="https://avatars.githubusercontent.com/u/62966167?v=4" width="150px;" alt="Eman Emad"/>
    <br />
    <sub><b>Eman Emad</b></sub></a>
    </td>
      </tr>