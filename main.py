from scipy.fft import fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pandas as pd
import copy
from PyQt5.QtWidgets import QSlider, QHBoxLayout, QLabel
import matplotlib as plt
from scipy.signal import butter, filtfilt, iirnotch
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
import os
import sys

plt.use('Qt5Agg')
import librosa
import bisect
import pyqtgraph as pg
from scipy import signal as sg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox


class SmoothingWindow:
    def __init__(self, window_type, amp, sigma=None):
        self.window_type = window_type
        self.amp = amp
        self.sigma = sigma

    def apply(self, signal):
        if self.window_type == "Rectangular":
            window = sg.windows.boxcar(len(signal))
            smoothed_signal = self.amp * window
            return smoothed_signal
        elif self.window_type == "Hamming":
            window = sg.windows.hamming(len(signal))
            smoothed_signal = self.amp * window
            return smoothed_signal
        elif self.window_type == "Hanning":
            window = sg.windows.hann(len(signal))
            smoothed_signal = self.amp * window
            return smoothed_signal
        elif self.window_type == "Gaussian":
            if self.sigma is not None:
                # Apply the Gaussian window to the signal with a specified standard deviation (sigma)
                window = sg.windows.gaussian(len(signal), self.sigma)
                smoothed_signal = self.amp * window
                return smoothed_signal
            else:
                raise ValueError("Gaussian window requires parameters.")


class Signal:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.time = []
        self.sample_rate = None
        self.freq_data = None
        self.Ranges = []
        self.phase = None


class CreateSlider:
    def __init__(self, index):
        # Create a slider
        self.index = index
        self.slider = QSlider()
        # sets the orientation of the slider to be vertical.
        self.slider.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(20)
        self.slider.setValue(10)
        # self.sliderlabel.setText()

    def get_slider(self):
        return self.slider


class AnimatedPlot:
    def __init__(self, plot_widget, data_x, data_y, update_interval=50):
        self.plot_widget = plot_widget
        self.data_x = data_x
        self.data_y = data_y
        self.current_index = 0
        self.plot_data = None
        self.timer = QTimer()
        self.base_interval = update_interval
        self.timer.setInterval(update_interval)
        self.timer.timeout.connect(self.update_plot)
        self.is_playing = False
        self.target_visualization_time = 4  # Target time in seconds to visualize any signal
        self.chunk_size = self.calculate_adaptive_chunk_size(len(data_x))
        self.speed = 1.0
        self.view_range = None

    def calculate_adaptive_chunk_size(self, total_points):
        """Calculate chunk size to complete visualization in target time"""
        # Calculate number of updates needed based on timer interval
        total_updates = (self.target_visualization_time * 1000) / self.base_interval
        # Calculate chunk size needed to show all points within target time
        chunk_size = max(100, int(total_points / total_updates))
        return chunk_size

    def initialize_plot(self):
        self.plot_widget.clear()
        self.current_index = 0
        self.plot_data = self.plot_widget.plot(pen='b')
        # Enable mouse interaction
        self.plot_widget.setMouseEnabled(x=True, y=True)

    def update_plot(self):
        if self.current_index >= len(self.data_x):
            self.timer.stop()
            self.is_playing = False
            return

        # Use adaptive chunk size
        end_index = min(self.current_index + int(self.chunk_size * self.speed), len(self.data_x))
        self.plot_data.setData(self.data_x[:end_index], self.data_y[:end_index])
        self.current_index = end_index
        self.is_playing = True

    def play(self):
        """Start or resume the animation"""
        if not self.is_playing:
            self.timer.start()
            self.is_playing = True

    def pause(self):
        """Pause the animation"""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False

    def reset(self):
        """Reset the animation to the beginning"""
        self.current_index = 0
        self.plot_data.setData([], [])
        self.timer.stop()
        self.is_playing = False

    def toggle_play_pause(self):
        """Toggle between play and pause states"""
        if self.is_playing == True:
            self.pause()
        else:
            self.play()

    def set_speed(self, speed):
        """Set the playback speed"""
        self.speed = speed
        self.timer.setInterval(int(self.base_interval / speed))

    def update_data(self, new_x, new_y):
        """Update the plot data"""
        self.data_x = new_x
        self.data_y = new_y
        # Recalculate chunk size for new data length
        self.chunk_size = self.calculate_adaptive_chunk_size(len(new_x))
        self.current_index = 0
        if self.is_playing:
            self.pause()
            self.play()

    def sync_view_range(self, view_range):
        """Synchronize view range with another plot"""
        self.view_range = view_range
        self.plot_widget.setXRange(*view_range[0], padding=0)
        self.plot_widget.setYRange(*view_range[1], padding=0)

    def get_view_range(self):
        """Get current view range"""
        return (
            self.plot_widget.getViewBox().viewRange()[0],
            self.plot_widget.getViewBox().viewRange()[1]
        )


class EqualizerApp(QtWidgets.QMainWindow):
    def __init__(self, lowcut=None, *args, **kwargs):
        super(EqualizerApp, self).__init__(*args, **kwargs)
        # Load the UI Page
        file_path = os.path.join(os.path.dirname(__file__), 'task3.ui')
        uic.loadUi(file_path, self)
        self.smoothing_window_combobox.hide()
        self.apply_btn.hide()
        self.label_7.hide()
        self.original_graph.setBackground("white")  # Fix color format
        self.equalized_graph.setBackground("white")
        self.frequancy_graph.setBackground("white")
        self.selected_mode = None
        # Initialize viewer settings
        self.setup_viewers()
        # self.setup_controls()

        # Connect signals for synchronization
        self.original_graph.getViewBox().sigRangeChanged.connect(self.sync_views_from_original)
        self.equalized_graph.getViewBox().sigRangeChanged.connect(self.sync_views_from_equalized)

        self.selected_window = None
        self.frame_layout = QHBoxLayout(self.sliders_frame)
        self.current_signal = None
        self.player = QMediaPlayer(None, QMediaPlayer.StreamPlayback)
        self.player.setVolume(50)
        self.timer = QTimer(self)
        self.timer = QtCore.QTimer(self)
        self.elapsed_timer = QtCore.QElapsedTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.updatepos)
        self.line = pg.InfiniteLine(pos=0, angle=90, pen=None, movable=False)
        self.changed_orig = False
        self.changed_eq = False
        self.player.positionChanged.connect(self.updatepos)
        self.current_speed = 1
        self.slider_gain = {}
        self.equalized_bool = False
        self.time_eq_signal = Signal('EqSignalInTime')
        self.eqsignal = None
        self.sampling_rate = None
        self.line = pg.InfiniteLine(pos=0.1, angle=90, pen=None, movable=False)
        self.original_animated_plot = None
        self.equalized_animated_plot = None
        # spectooooooo
        self.available_palettes = ['twilight', 'Blues', 'Greys', 'ocean', 'nipy_spectral']
        self.current_color_palette = self.available_palettes[0]
        self.spectrogram_widget = {
            'before': self.spectrogram_before,
            'after': self.spectrogram_after
        }
        # Ui conection
        self.modes_combobox.activated.connect(lambda: self.combobox_activated())
        self.smoothing_window_combobox.activated.connect(lambda: self.smoothing_window_combobox_activated())
        self.lineEdit_2.setVisible(False)  # Initially hide the line edit for Gaussian window
        self.load_btn.clicked.connect(lambda: self.load())
        self.hear_orig_btn.clicked.connect(lambda: self.playMusic('orig'))
        self.hear_eq_btn.clicked.connect(lambda: self.playMusic('equalized'))
        # Modify existing button connections
        # Modify existing button connections
        self.play_pause_btn.clicked.connect(self.toggle_animation)
        self.replay_btn.clicked.connect(self.reset_animation)
        self.speed_up_btn.clicked.connect(self.speed_up_animation)
        self.speed_down_btn.clicked.connect(self.speed_down_animation)
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_in())
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_out())
        self.checkBox.stateChanged.connect(lambda: self.hide())
        self.dictionary = {
            'Uniform Range': {},

            "Animals and Instruments Sounds": {"Dog": [300, 1300],
                                               "Tiger": [100, 300],
                                               "Birds": [2000, 6000],
                                               "Guitar": [82, 300],
                                               "Flute": [6000, 16000],
                                               "Xylophone": [1300, 2000]

                                               },
            'Vowels and Music': {
                # 'We' vowel: Strong peak around 300-500 Hz
                "We": [300, 500],  # First major peak in spectrum
                
                # 'Rock': Multiple peaks between 1000-3000 Hz
                "Rock": [1000, 3000],  # Multiple peaks in mid-range
                
                # Electric Guitar: Low-frequency content 100-300 Hz
                "Guitar": [100, 300],  # Lower frequency region
                
                # Clapping/Stomping: High-frequency peaks 3000-9000 Hz
                "Stomp/Clap": [3000, 9000]  # Higher frequency content
            },
            'Wiener Filter': {}  # Add new mode
        }
        # Add new instance variables
        self._cached_fft = None
        self._cached_freqs = None
        self.cached = False
        self.noise_segment = None
        self.noise_region = None
        self.region_item = None
        self.is_selecting_noise = False
        # ddffffffff
        
        #hghhf
    from scipy.signal import butter, filtfilt
    from scipy.io.wavfile import read

  #  sample_rate, audio_data = read(r"C:\Users\youssef elawady\Downloads\Gotye - Somebody That I Used To Know (feat. Kimbra) [Official Music Video] (online-video-cutter.com).wav")

    def highpass_filter(data, cutoff, sample_rate, order=4):
        """
        Apply a highpass filter to remove low-frequency components (e.g., music).
        """
        nyquist = 0.5 * sample_rate
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='highpass')
        y = filtfilt(b, a, data)
        return y

    def bandstop_filter(data, lowcut, highcut, sample_rate, order=4):
        """
        Apply a bandstop filter to remove specific frequency ranges (e.g., music harmonics).
        """
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='bandstop')
        y = filtfilt(b, a, data)
        return y

    def remove_music(data, sample_rate):
        """
        Apply a combination of filters to suppress music frequencies.
        """
        # Remove low frequencies (e.g., bass, drum beats)
        filtered_data = data.highpass_filter(data, cutoff=300, sample_rate=sample_rate, order=4)

        # Remove harmonics and instrument sounds (e.g., piano, guitar)
        music_harmonics = [(300, 1000), (1000, 4000)]  # Example frequency ranges
        for lowcut, highcut in music_harmonics:
            filtered_data = data.bandstop_filter(filtered_data, lowcut, highcut, sample_rate, order=4)

        return filtered_data

    # Example usage

        # Remove music
        filtered_audio = remove_music(audio_data, sample_rate)

        # Save the output audio
        write("output_audio_no_music.wav", sample_rate, filtered_audio.astype(np.int16))

        # Plot the original and filtered signals
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title("Original Signal")
        plt.plot(audio_data)
        plt.subplot(2, 1, 2)
        plt.title("Filtered Signal (Music Removed)")
        plt.plot(filtered_audio)
        plt.tight_layout()
        plt.show()

    #dd


    def setup_viewers(self):
        # Setup viewers with white backgrounds
        self.original_graph.setBackground("white")
        self.equalized_graph.setBackground("white")
        self.frequancy_graph.setBackground("white")

        # Enable mouse interactions
        for graph in [self.original_graph, self.equalized_graph]:
            graph.setMouseEnabled(x=True, y=True)
            graph.setMenuEnabled(False)

        # Setup spectrograms
        self.setup_spectrograms()
        # Initialize region selection tool but don't add it yet
        self.region_item = None

    def setup_controls(self):
        # Connect control buttons
        self.play_pause_btn.clicked.connect(self.toggle_animation)
        self.replay_btn.clicked.connect(self.reset_animation)
        self.speed_up_btn.clicked.connect(self.speed_up_animation)
        self.speed_down_btn.clicked.connect(self.speed_down_animation)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)

    def sync_views_from_original(self):
        if not self.original_animated_plot:
            return
        view_range = self.original_animated_plot.get_view_range()
        if self.equalized_animated_plot:
            self.equalized_animated_plot.sync_view_range(view_range)

    def sync_views_from_equalized(self):
        if not self.equalized_animated_plot:
            return
        view_range = self.equalized_animated_plot.get_view_range()
        if self.original_animated_plot:
            self.original_animated_plot.sync_view_range(view_range)

    # OUR CODE HERE
    def load(self):
        path_info = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select a signal...", os.getenv('HOME'), filter="Raw Data (*.csv *.wav *.mp3)")
        path = path_info[0]
        # print(path)
        time = []
        self.equalized_bool = False
        sample_rate = 0
        data = []
        signal_name = path.split('/')[-1].split('.')[0]  # Extract signal name from file path
        type = path.split('.')[-1]
        # Check the file type and load data accordingly
        if type in ["wav", "mp3"]:
            data, sample_rate = librosa.load(path)
            Duration = librosa.get_duration(y=data, sr=sample_rate)
            self.duration = Duration
            time = np.linspace(0, Duration, len(data))
            self.audio_data = path
        elif type == "csv":
            data_of_signal = pd.read_csv(path)
            time = np.array(data_of_signal.iloc[:, 0].astype(float).tolist())
            data = np.array(data_of_signal.iloc[:, 1].astype(float).tolist())
            if len(time) > 1:
                sample_rate = 1 / (time[1] - time[0])
                sample_rate = 1 / (time[1] - time[0])
            else:
                sample_rate = 1
        # Create a Signal object and set its attributes
        self.current_signal = Signal(signal_name)
        self.current_signal.data = data
        self.current_signal.time = time
        self.current_signal.sample_rate = sample_rate
        self.sampling_rate = sample_rate
        # Calculate and set the Fourier transform of the signal
        T = 1 / self.current_signal.sample_rate
        x_data, y_data = self.get_Fourier(T, self.current_signal.data)
        self.current_signal.freq_data = [x_data, y_data]
        for i in range(10):
            self.batch_size = len(self.current_signal.freq_data[0]) // 10
            self.dictionary['Uniform Range'][i] = [i * self.batch_size, (i + 1) * self.batch_size]
            # selected_index = None
        # self.add_slider(selected_index)
        self.frequancy_graph.clear()
        if self.spectrogram_after.count() > 0:
            # If yes, remove the existing canvas
            self.spectrogram_after.itemAt(0).widget().setParent(None)
        self.Plot("original")
        self.plot_spectrogram(data, sample_rate, self.spectrogram_before)
        self.frequancy_graph.plot(self.current_signal.freq_data[0],
                                  self.current_signal.freq_data[1], pen={'color': 'b'})
        self.eqsignal = copy.deepcopy(self.current_signal)
        # Initialize synchronized animated plots
        self.original_animated_plot = AnimatedPlot(
            self.original_graph,
            self.current_signal.time,
            self.current_signal.data
        )
        self.original_animated_plot.initialize_plot()

        self.equalized_animated_plot = AnimatedPlot(
            self.equalized_graph,
            self.current_signal.time,
            self.current_signal.data
        )
        self.equalized_animated_plot.initialize_plot()

        # Ensure initial view synchronization
        self.sync_views_from_original()

        # Start both animations
        self.original_animated_plot.play()
        self.equalized_animated_plot.play()

    def toggle_animation(self):
        if self.original_animated_plot and self.equalized_animated_plot:
            self.original_animated_plot.toggle_play_pause()
            self.equalized_animated_plot.toggle_play_pause()

    def reset_animation(self):
        if self.original_animated_plot and self.equalized_animated_plot:
            self.original_animated_plot.reset()
            self.equalized_animated_plot.reset()
            self.original_animated_plot.play()
            self.equalized_animated_plot.play()
            print("replaaaaaay")

    def speed_up_animation(self):
        if self.original_animated_plot and self.equalized_animated_plot:
            self.current_speed = min(4.0, self.current_speed + 0.1)
            self.original_animated_plot.set_speed(self.current_speed)
            self.equalized_animated_plot.set_speed(self.current_speed)
            print("speeeeed up")

    def speed_down_animation(self):
        if self.original_animated_plot and self.equalized_animated_plot:
            self.current_speed = max(0.1, self.current_speed - 0.1)
            self.original_animated_plot.set_speed(self.current_speed)
            self.equalized_animated_plot.set_speed(self.current_speed)
            print("speeeeeeed down")

    def setup_spectrograms(self):
        self.spectrogram_widget = {
            'before': self.spectrogram_before,
            'after': self.spectrogram_after
        }
        # Connect spectrogram visibility toggle
        self.checkBox.stateChanged.connect(self.toggle_spectrograms)

    def toggle_spectrograms(self):
        visible = not self.checkBox.isChecked()
        self.specto_frame_before.setVisible(visible)
        self.specto_frame_after.setVisible(visible)
        self.label_3.setVisible(visible)
        self.label_4.setVisible(visible)

    def get_Fourier(self, T, data):
        N = len(data)
        freq_amp = np.fft.fft(data)
        self.current_signal.phase = np.angle(freq_amp[:N // 2])
        # Calculate the corresponding frequencies
        Freq = np.fft.fftfreq(N, T)[:N // 2]
        # Extracting positive frequencies and scaling the amplitude
        Amp = (2 / N) * (np.abs(freq_amp[:N // 2]))
        return Freq, Amp

    def Range_spliting(self):
        # Guard clause for no signal
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            print("Warning: No signal loaded")
            return

        # Guard clause for no frequency data
        if not hasattr(self.current_signal, 'freq_data') or self.current_signal.freq_data is None:
            print("Warning: No frequency data available")
            return

        # Initialize Ranges if it doesn't exist
        if not hasattr(self.current_signal, 'Ranges'):
            self.current_signal.Ranges = []
        else:
            # Clear existing ranges
            self.current_signal.Ranges = []

        freq = self.current_signal.freq_data[0]  # Index zero for values of freq

        try:
            if self.selected_mode == 'Uniform Range':
                self.batch_size = len(freq) // 10
                self.current_signal.Ranges = [(i * self.batch_size, (i + 1) * self.batch_size) for i in range(10)]
            else:
                dict_ranges = self.dictionary[self.selected_mode]
                for _, (start, end) in dict_ranges.items():
                    start_ind = bisect.bisect_left(freq, start)
                    end_ind = bisect.bisect_right(freq, end)
                    if start_ind < len(freq) and end_ind <= len(freq):
                        self.current_signal.Ranges.append((start_ind, end_ind))

            # Copy ranges to eqsignal if it exists
            if hasattr(self, 'eqsignal'):
                self.eqsignal.Ranges = copy.deepcopy(self.current_signal.Ranges)

        except Exception as e:
            print(f"Error in Range_spliting: {str(e)}")
            # Reset ranges to empty list in case of error
            self.current_signal.Ranges = []
            if hasattr(self, 'eqsignal'):
                self.eqsignal.Ranges = []

    def add_slider(self, selected_index):
        # Guard clause for no signal
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            print("Warning: No signal loaded")
            return

        # Clear existing sliders
        self.clear_layout(self.frame_layout)

        # Reset slider gain dictionary
        self.slider_gain = {}

        # Initialize ranges first
        self.Range_spliting()

        # Debug print to check mode name
        print(f"Selected mode: '{selected_index}'")
        print(f"Available modes: {list(self.dictionary.keys())}")

        # Guard clause for invalid mode
        if selected_index not in self.dictionary:
            print(f"Warning: Invalid mode {selected_index}")
            return

        try:
            dictionary = self.dictionary[selected_index]
            for i, (key, _) in enumerate(dictionary.items()):
                # Create label and slider
                label = QLabel(str(key))
                slider_creator = CreateSlider(i)
                slider = slider_creator.get_slider()

                # Initialize gain for this slider
                self.slider_gain[i] = 10

                # Connect slider value change event with a fix for the closure issue
                def make_callback(idx):
                    return lambda value: self.update_slider_value(idx, value / 10)

                slider.valueChanged.connect(make_callback(i))

                # Add widgets to layout
                self.frame_layout.addWidget(slider)
                self.frame_layout.addWidget(label)

        except Exception as e:
            print(f"Error creating sliders: {str(e)}")
            import traceback
            traceback.print_exc()

    def equalized(self, slider_index, value):
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            print("Warning: No signal loaded")
            return

        if not self.current_signal.Ranges or slider_index >= len(self.current_signal.Ranges):
            print(f"Warning: Invalid slider index {slider_index}")
            return

        self.equalized_bool = True

        # Initialize time_eq_signal if needed
        if not hasattr(self, 'time_eq_signal'):
            self.time_eq_signal = Signal('EqSignalInTime')

        self.time_eq_signal.time = self.current_signal.time

        # Get smoothing window parameters
        windowtype = self.smoothing_window_combobox.currentText()
        sigma_text = self.lineEdit_2.text()
        sigma = int(sigma_text) if sigma_text else 20

        try:
            start, end = self.current_signal.Ranges[slider_index]

            # For vowel removal, we want to attenuate the frequencies
            # A value less than 1 will reduce the amplitude of these frequencies
            if self.selected_mode == 'Vowels and Music':
                value = min(value, 0.3)  # Limit maximum attenuation for vowels

            # Apply the smoothing window
            smooth_window = SmoothingWindow(windowtype, 1, sigma)
            curr_smooth_window = smooth_window.apply(self.current_signal.freq_data[1][start:end])
            curr_smooth_window *= value

            Amp = np.array(self.current_signal.freq_data[1][start:end])
            new_amp = Amp * curr_smooth_window
            self.eqsignal.freq_data[1][start:end] = new_amp

            # Update plots and spectrograms
            self.plot_freq_smoothing_window()
            self.time_eq_signal.data = self.recovered_signal(self.eqsignal.freq_data[1], self.current_signal.phase)

            # Adjust time array length to match data length
            excess = len(self.time_eq_signal.time) - len(self.time_eq_signal.data)
            if excess > 0:
                self.time_eq_signal.time = self.time_eq_signal.time[:-excess]

            self.Plot("equalized")
            self.plot_spectrogram(self.time_eq_signal.data, self.current_signal.sample_rate, self.spectrogram_after)

            # Update animated plot if it exists
            if self.equalized_animated_plot:
                self.equalized_animated_plot.update_data(
                    self.time_eq_signal.time,
                    self.time_eq_signal.data
                )
                self.sync_views_from_original()

        except Exception as e:
            print(f"Error in equalized: {str(e)}")
            return

    def Plot(self, graph):
        signal = self.time_eq_signal if self.equalized_bool else self.current_signal
        if signal:
            # time domain
            self.equalized_graph.clear()
            graphs = [self.original_graph, self.equalized_graph]
            graph = graphs[0] if graph == "original" else graphs[1]
            graph.clear()
            graph.setLabel('left', "Amplitude")
            graph.setLabel('bottom', "Time")
            plot_item = graph.plot(
                signal.time, signal.data, name=f"{signal.name}")
            # Add legend to the graph
            if graph.plotItem.legend is not None:
                graph.plotItem.legend.clear()
            legend = graph.addLegend()
            legend.addItem(plot_item, name=f"{signal.name}")

    def plot_freq_smoothing_window(self):
        signal = self.eqsignal if self.equalized_bool else self.current_signal
        if signal and signal.Ranges:
            _, end_last_ind = signal.Ranges[-1]
            self.frequancy_graph.clear()
            
            # Plot the modified frequency data
            self.frequancy_graph.plot(signal.freq_data[0][:end_last_ind],
                                    signal.freq_data[1][:end_last_ind], 
                                    pen={'color': 'b', 'width': 1},
                                    name='Frequency Response')

            # Highlight the modified frequency ranges
            for i, (start_ind, end_ind) in enumerate(signal.Ranges):
                # Get the current gain for this range from slider
                current_gain = self.slider_gain.get(i, 1.0)
                
                # Plot the modified frequency range with different color
                freq_range = signal.freq_data[0][start_ind:end_ind]
                amp_range = signal.freq_data[1][start_ind:end_ind]
                
                # Highlight modified ranges
                self.frequancy_graph.plot(freq_range, amp_range,
                                        pen={'color': 'r', 'width': 2})

                # Add vertical lines to mark range boundaries
                start_line = signal.freq_data[0][start_ind]
                end_line = signal.freq_data[0][end_ind - 1]
                v_line_start = pg.InfiniteLine(pos=start_line, angle=90, 
                                            movable=False, 
                                            pen=pg.mkPen('r', width=1))
                v_line_end = pg.InfiniteLine(pos=end_line, angle=90, 
                                        movable=False, 
                                        pen=pg.mkPen('r', width=1))
                self.frequancy_graph.addItem(v_line_start)
                self.frequancy_graph.addItem(v_line_end)

    def plot_spectrogram(self, samples, sampling_rate, widget):
        if widget.count() > 0:
            # If yes, remove the existing canvas
            widget.itemAt(0).widget().setParent(None)
        data = samples.astype('float32')
        # Size of the Fast Fourier Transform (FFT), which will also be used as the window length
        n_fft = 500
        # Step or stride between windows. If the step is smaller than the window length, the windows will overlap
        hop_length = 320
        window_type = 'hann'
        # Compute the short-time Fourier transform magnitude squared
        frequency_magnitude = np.abs(
            librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type)) ** 2
        # Compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(S=frequency_magnitude, y=data, sr=sampling_rate, n_fft=n_fft,
                                                         hop_length=hop_length, win_length=n_fft, window=window_type,
                                                         n_mels=128)
        # Convert power spectrogram to decibels
        decibel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        time_axis = np.linspace(0, len(data) / sampling_rate)
        fig = Figure()
        fig = Figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.imshow(decibel_spectrogram, aspect='auto', cmap='viridis',
                  extent=[time_axis[0], time_axis[-1], 0, sampling_rate / 2])
        ax.axes.plot()
        canvas = FigureCanvas(fig)
        widget.addWidget(canvas)

    def replay(self):
        if self.type == 'orig':
            self.playMusic('orig')
        else:
            self.playMusic('equalized')

    def playMusic(self, type):
        self.current_speed = 1
        self.line_position = 0
        self.player.setPlaybackRate(self.current_speed)
        media = QMediaContent(QUrl.fromLocalFile(self.audio_data))
        # Set the media content for the player and start playing
        self.player.setMedia(media)
        self.type = type
        if type == 'orig':
            sd.stop()
            self.timer.stop()
            self.changed_orig = True
            self.changed_eq = False
            # Create a QMediaContent object from the local audio file
            self.player.play()
            self.player.setVolume(100)
            # Add a vertical line to the original graph
            self.equalized_graph.removeItem(self.line)
            self.original_graph.addItem(self.line)
            self.timer.start()
        else:
            self.changed_eq = True
            self.changed_orig = False
            self.timer.start()
            self.player.play()
            self.player.setVolume(0)
            self.original_graph.removeItem(self.line)
            self.equalized_graph.addItem(self.line)
            sd.play(self.time_eq_signal.data, self.current_signal.sample_rate, blocking=False)
            self.player.play()

    def updatepos(self):
        max_x = self.original_graph.getViewBox().viewRange()[0][1]
        graphs = [self.original_graph, self.equalized_graph]
        graph = graphs[0] if self.changed_orig else graphs[1]
        # Get the current position in milliseconds
        position = self.player.position() / 1000
        # Update the line position based on the current position
        self.line_position = position
        max_x = graph.getViewBox().viewRange()[0][1]
        # print(position)
        if self.line_position > max_x:
            self.line_position = max_x
        self.line_position = position
        self.line.setPos(self.line_position)

    def replay(self):
        if self.type == 'orig':
            self.playMusic('orig')
            print("replay")
        else:
            self.playMusic('equalized')
            print("replay")

    def play_pause(self):
        if self.changed_orig:
            self.player.pause()
            AnimatedPlot.timer.stop()
            self.changed_orig = not self.changed_orig
            print("stop")
        else:
            self.player.play()
            AnimatedPlot.timer.start()
            self.changed_orig = not self.changed_orig

    def combobox_activated(self):
        # First check if a signal is loaded
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            # Show warning to user that they need to load a signal first
            QtWidgets.QMessageBox.warning(
                self,
                "No Signal Loaded",
                "Please load a signal before selecting a mode.",
                QtWidgets.QMessageBox.Ok
            )
            # Reset combobox to first item
            self.modes_combobox.setCurrentIndex(0)
            return

        # If we have a signal, proceed normally
        selected_index = self.modes_combobox.currentIndex()
        self.selected_mode = self.modes_combobox.currentText()
        if self.selected_mode == 'Wiener Filter':
            # Hide sliders for Wiener filter mode
            self.clear_layout(self.frame_layout)
            
            # Add select noise region button
            select_noise_btn = QtWidgets.QPushButton("Select Noise Region")
            select_noise_btn.clicked.connect(self.toggle_noise_selection)
            select_noise_btn.setMinimumHeight(40)
            select_noise_btn.setStyleSheet("""
                QPushButton {
                    background: rgb(0, 111, 163);
                    border-radius: 10px;
                    color: white;
                }
                QPushButton:hover {
                    background-color: white;
                    border: 2px solid #475c7a;
                    border-radius: 10px;
                    color: #34455e;
                }
            """)
            self.frame_layout.addWidget(select_noise_btn)
            
            # Add apply button for Wiener filter
            apply_button = QtWidgets.QPushButton("Apply Wiener Filter")
            apply_button.clicked.connect(self.apply_wiener_filter)
            apply_button.setMinimumHeight(40)
            apply_button.setStyleSheet("""
                QPushButton {
                    background: rgb(0, 111, 163);
                    border-radius: 10px;
                    color: white;
                }
                QPushButton:hover {
                    background-color: white;
                    border: 2px solid #475c7a;
                    border-radius: 10px;
                    color: #34455e;
                }
            """)
            self.frame_layout.addWidget(apply_button)
        else:
            # Original slider-based modes
            self.add_slider(self.selected_mode)
            self.Range_spliting()

    def apply_wiener_filter(self):
        """Apply Wiener filter to the signal"""
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            return

        if self.noise_region is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Noise Region",
                "Please select a noise region first.",
                QtWidgets.QMessageBox.Ok
            )
            return

        # Get or calculate FFT
        if not self.cached:
            signal_fft = np.fft.fft(self.current_signal.data)
            self._cached_freqs = np.fft.fftfreq(len(self.current_signal.data), 
                                               1/self.current_signal.sample_rate)
            self._cached_fft = signal_fft
            self.cached = True

        # Estimate noise from the first 1000 samples (or another quiet section)
        # You might want to make this configurable or use a better noise estimation method
        self.noise_segment = self.current_signal.data[:1000]
        
        # Apply Wiener filter
        try:
            # Compute noise and signal power spectral densities using selected region
            noise_fft = np.fft.fft(self.noise_region, n=len(self.current_signal.data))
            noise_psd = np.abs(noise_fft)**2
            signal_psd = np.abs(self._cached_fft)**2

            # Compute Wiener filter
            wiener_filter = signal_psd / (signal_psd + noise_psd)
            modified_fft = self._cached_fft * wiener_filter

            # Apply inverse FFT to get filtered signal
            filtered_signal = np.real(np.fft.ifft(modified_fft))

            # Update equalized signal
            self.equalized_bool = True
            self.time_eq_signal.data = filtered_signal
            self.time_eq_signal.time = self.current_signal.time[:len(filtered_signal)]

            # Update plots
            self.Plot("equalized")
            self.plot_spectrogram(filtered_signal, self.current_signal.sample_rate, 
                                self.spectrogram_after)

            # Update animated plot
            if self.equalized_animated_plot:
                self.equalized_animated_plot.update_data(
                    self.time_eq_signal.time,
                    self.time_eq_signal.data
                )
                self.sync_views_from_original()

        except Exception as e:
            print(f"Error applying Wiener filter: {str(e)}")
            import traceback
            traceback.print_exc()

    def toggle_noise_selection(self):
        """Toggle noise region selection mode"""
        try:
            self.is_selecting_noise = not self.is_selecting_noise
            
            if self.is_selecting_noise:
                # Create region selection tool if it doesn't exist
                if self.region_item is None:
                    self.region_item = pg.LinearRegionItem([0, 0], movable=True, brush=(50, 50, 200, 50))
                    self.original_graph.addItem(self.region_item)
                
                # Show region selection tool
                if self.current_signal is not None:
                    # Set initial region to middle 20% of signal
                    signal_len = len(self.current_signal.time)
                    start_idx = int(signal_len * 0.4)
                    end_idx = int(signal_len * 0.6)
                    start_time = self.current_signal.time[start_idx]
                    end_time = self.current_signal.time[end_idx]
                    self.region_item.setRegion([start_time, end_time])
                    self.region_item.show()
            else:
                # Hide region selection tool and store selected region
                if self.region_item is not None:
                    start, end = self.region_item.getRegion()
                    # Convert time to indices
                    start_idx = np.searchsorted(self.current_signal.time, start)
                    end_idx = np.searchsorted(self.current_signal.time, end)
                    self.noise_region = self.current_signal.data[start_idx:end_idx]
                    self.region_item.hide()

        except Exception as e:
            print(f"Error in toggle_noise_selection: {str(e)}")
            import traceback
            traceback.print_exc()

    def smoothing_window_combobox_activated(self):
        selected_item = self.smoothing_window_combobox.currentText()
        self.selected_window = selected_item
        # Show or hide the line edit based on the selected smoothing window
        self.lineEdit_2.setVisible(selected_item == 'Gaussian')

    def clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

    def update_slider_value(self, slider_index, value):
        # This method will be called whenever a slider is moved
        self.slider_gain[slider_index] = value
        # print (self.slider_gain)
        self.equalized(slider_index, value)
        # self.Plot('equalized')

    def zoom_in(self):
        self.original_graph.getViewBox().scaleBy((0.8, 0.8))
        self.equalized_graph.getViewBox().scaleBy((0.8, 0.8))
        print('zoomed in')

    def zoom_out(self):
        self.original_graph.getViewBox().scaleBy((1.2, 1.2))
        self.equalized_graph.getViewBox().scaleBy((1.2, 1.2))
        print('zoomed out')

    def recovered_signal(self, Amp, phase):
        # complex array from amp and phase comination
        Amp = Amp * len(
            self.current_signal.data) / 2  # N/2 as we get amp from foureir by multiplying it with fraction 2/N
        complex_value = Amp * np.exp(1j * phase)
        # taking inverse fft to get recover signal
        recovered_signal = np.fft.irfft(complex_value)
        # taking only the real part of the signal
        return (recovered_signal)

    def hide(self):
        if (self.checkBox.isChecked()):
            self.specto_frame_before.hide()
            self.label_3.setVisible(False)
            self.specto_frame_after.hide()
            self.label_4.setVisible(False)
        else:
            self.specto_frame_before.show()
            self.label_3.setVisible(True)
            self.specto_frame_after.show()
            self.label_4.setVisible(True)


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = EqualizerApp()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


















