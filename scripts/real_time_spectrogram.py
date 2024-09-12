import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import spectrogram
from scipy.signal.windows import dpss, hann, gaussian, hamming

# Parameters
rate = 48000
default_time_frame = 2  # Default spectrogram time frame in seconds
buffer_size = rate * default_time_frame  # Buffer size is 2 seconds of audio data
audio_buffer = np.zeros(buffer_size)  # Buffer to hold the most recent audio data

class SpectrogramGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-time Spectrogram")
        self.geometry("1000x800")

        # Create a frame for the controls on the left
        controls_frame = tk.Frame(self)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y)

        # FFT samples setting
        tk.Label(controls_frame, text="FFT Samples").pack()
        self.fft_samples_entry = tk.Entry(controls_frame)
        self.fft_samples_entry.pack()
        self.fft_samples_entry.insert(tk.END, 2048)  # Default FFT points

        # Overlap setting
        tk.Label(controls_frame, text="Overlap").pack()
        self.overlap_entry = tk.Entry(controls_frame)
        self.overlap_entry.pack()
        self.overlap_entry.insert(tk.END, 1568)  # Default overlap

        # Time frame setting
        tk.Label(controls_frame, text="Time Frame (s)").pack()
        self.time_frame_entry = tk.Entry(controls_frame)
        self.time_frame_entry.pack()
        self.time_frame_entry.insert(tk.END, default_time_frame)

        # Frequency scale setting
        tk.Label(controls_frame, text="Max Frequency (Hz)").pack()
        self.max_freq_entry = tk.Entry(controls_frame)
        self.max_freq_entry.pack()
        self.max_freq_entry.insert(tk.END, 5000)  # Default max frequency (Nyquist frequency) can be rate // 2

        # Colormap setting
        tk.Label(controls_frame, text="Color Map").pack()
        self.color_map_var = tk.StringVar()
        color_map_menu = ttk.Combobox(controls_frame, textvariable=self.color_map_var)
        color_map_menu['values'] = ('viridis', 'plasma', 'inferno', 'magma', 'cividis')
        color_map_menu.set('inferno')  # Default colormap
        color_map_menu.pack()

        # Color scale min and max setting
        tk.Label(controls_frame, text="Color Scale Min (dB)").pack()
        self.color_min_entry = tk.Entry(controls_frame)
        self.color_min_entry.pack()
        self.color_min_entry.insert(tk.END, 0)  # Default color scale min

        tk.Label(controls_frame, text="Color Scale Max (dB)").pack()
        self.color_max_entry = tk.Entry(controls_frame)
        self.color_max_entry.pack()
        self.color_max_entry.insert(tk.END, 50)  # Default color scale max

        # Start, pause, and resume buttons
        self.start_button = tk.Button(controls_frame, text="Start", command=self.start_spectrogram)
        self.start_button.pack()

        self.pause_button = tk.Button(controls_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack()

        # Create the plot on the right
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        # Initialize some variables
        self.pcolormesh = None  # Handle for the pcolormesh plot
        self.updating = False
        self.stream = None

        # Bind events to update parameters dynamically
        self.fft_samples_entry.bind("<Return>", self.update_parameters)
        self.overlap_entry.bind("<Return>", self.update_parameters)
        self.time_frame_entry.bind("<Return>", self.update_parameters)
        self.max_freq_entry.bind("<Return>", self.update_parameters)
        self.color_min_entry.bind("<Return>", self.update_parameters)
        self.color_max_entry.bind("<Return>", self.update_parameters)
        color_map_menu.bind("<<ComboboxSelected>>", self.update_parameters)

    def update_parameters(self, event=None):
        """Update spectrogram parameters immediately without pausing."""
        if self.stream:
            # Get the new parameters from the GUI
            fft_points = int(self.fft_samples_entry.get())
            overlap = int(self.overlap_entry.get())
            time_frame = float(self.time_frame_entry.get())
            max_frequency = int(self.max_freq_entry.get())
            colormap = self.color_map_var.get()
            color_min = int(self.color_min_entry.get())
            color_max = int(self.color_max_entry.get())

            # Adjust buffer size according to time frame
            global buffer_size, audio_buffer
            buffer_size = int(rate * time_frame)
            audio_buffer = np.zeros(buffer_size)

            # Update the spectrogram loop with the new parameters
            self.update_spectrogram_loop(fft_points, overlap, colormap, max_frequency, color_min, color_max)

    def start_spectrogram(self):
        """Start capturing and plotting the live audio spectrogram."""
        # Get the initial parameters from the GUI
        fft_points = int(self.fft_samples_entry.get())
        overlap = int(self.overlap_entry.get())
        time_frame = float(self.time_frame_entry.get())
        max_frequency = int(self.max_freq_entry.get())
        colormap = self.color_map_var.get()
        color_min = int(self.color_min_entry.get())
        color_max = int(self.color_max_entry.get())

        # Set buffer size according to time frame
        global buffer_size, audio_buffer
        buffer_size = int(rate * time_frame)
        audio_buffer = np.zeros(buffer_size)

        # Enable the pause button and disable the start button
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)

        self.updating = True

        # Start the audio stream
        self.stream = start_audio_stream(fft_points, overlap)

        # Begin updating the spectrogram
        self.update_spectrogram_loop(fft_points, overlap, colormap, max_frequency, color_min, color_max)

    def toggle_pause(self):
        """Pause or resume the spectrogram."""
        self.updating = not self.updating
        if self.updating:
            self.pause_button.config(text="Pause")
            # Resume the spectrogram loop when unpaused
            self.update_parameters()
        else:
            self.pause_button.config(text="Resume")

    def update_spectrogram_loop(self, fft_points, overlap, colormap, max_frequency, color_min, color_max):
        """Update the spectrogram in real-time based on audio input."""
        global audio_buffer
        #NW = 4

        #tapers = dpss(M = fft_points, NW = NW ,Kmax=4)
        hanwin = hann(M = fft_points)
        #gauswin = gaussian(M = fft_points, std=fft_points/2)
        #hamwin = hamming(M = fft_points)

        
        if self.updating:
            # Compute the spectrogram from the accumulated audio data
            f, t, Sxx = spectrogram(audio_buffer, fs=rate, nperseg=fft_points, noverlap=overlap,window=hanwin)
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)
            #frequencies, times, Sxx_dpss_1 = spectrogram(audio_buffer, fs=rate, window = tapers[0], nperseg = fft_points, noverlap = overlap, nfft=fft_points, mode = "complex")
            #frequencies, times, Sxx_dpss_2 = spectrogram(audio_buffer, fs=rate, window = tapers[1], nperseg = fft_points, noverlap = overlap,nfft=fft_points, mode = "complex")
            #frequencies, times, Sxx_dpss_3 = spectrogram(audio_buffer, fs=rate, window = tapers[2], nperseg = fft_points, noverlap = overlap,nfft=fft_points, mode = "complex")
            #frequencies, times, Sxx_dpss_4 = spectrogram(audio_buffer, fs=rate, window = tapers[3], nperseg = fft_points, noverlap = overlap,nfft=fft_points, mode = "complex")
            
            #Sxx_dpss = np.real(Sxx_dpss_1*np.conj(Sxx_dpss_1) + 
            #                   Sxx_dpss_2*np.conj(Sxx_dpss_2) +
            #                   Sxx_dpss_3*np.conj(Sxx_dpss_3))
            #Sxx_dB = 10 * np.log10(Sxx_dpss + 1e-10)
            # Remove the previous pcolormesh if it exists
            if self.pcolormesh is not None:
                self.pcolormesh.remove()

            # Reset the axes to fit the new time frame and frequency range
            self.ax.set_xlim(0, t.max())
            self.ax.set_ylim(0, max_frequency)

            # Redraw the pcolormesh with updated data
            self.pcolormesh = self.ax.pcolormesh(t, f, Sxx_dB, cmap=colormap, vmin=color_min, vmax=color_max)
            self.ax.set_ylabel('Frequency [Hz]')
            self.ax.set_xlabel('Time [sec]')
            self.ax.set_ylim(0, max_frequency)

            # Redraw the canvas
            self.canvas.draw()

            # Use `self.after` to check for the next buffer
            self.after(1, self.update_spectrogram_loop, fft_points, overlap, colormap, max_frequency, color_min, color_max)

def start_audio_stream(fft_points, overlap):
    """Start the PyAudio stream to capture live audio."""
    p = pyaudio.PyAudio()

    # Callback to fill the audio buffer
    def callback(in_data, frame_count, time_info, status):
        global audio_buffer
        new_audio_data = np.frombuffer(in_data, dtype=np.int16)

        # Roll the buffer to discard the oldest audio and append the new audio data
        audio_buffer = np.roll(audio_buffer, -len(new_audio_data))
        audio_buffer[-len(new_audio_data):] = new_audio_data

        return (in_data, pyaudio.paContinue)

    # Open a non-blocking stream to capture audio input
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=fft_points,
                    stream_callback=callback)

    stream.start_stream()
    return stream

# Create and run the GUI application
app = SpectrogramGUI()
app.mainloop()
