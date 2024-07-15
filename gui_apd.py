import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import signal
import platform
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

class Application(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.pack_propagate(0)  # Ensure the frame size is controlled by the window size
        self.create_widgets()  # Create the GUI widgets
        self.process_detection = None  # Initialize the detection detection process variable

    def create_widgets(self):
        # Create a style for the Labelframe
        style = ttk.Style()
        style.configure("Custom.TLabelframe.Label", foreground="white")
        
        # detection Detection Section
        # Create a labeled frame for detection Detection with custom style
        self.detection_frame = ttk.Labelframe(self, text="People Detection", bootstyle="primary", style="Custom.TLabelframe")
        self.detection_frame.pack(fill=tk.X, padx=10, pady=5)
        # Create and pack the start button for detection Detection
        self.play_button_detection = ttk.Button(self.detection_frame, text="Start People Detection", command=self.start_program_detection, bootstyle="info", width=20)
        self.play_button_detection.pack(pady=5)
        # Create and pack the stop button for detection Detection
        self.stop_button_detection = ttk.Button(self.detection_frame, text="Stop People Detection", command=self.stop_program_detection, bootstyle="danger", width=20)
        self.stop_button_detection.pack(pady=5)

        # Source parameter
        self.source_label = ttk.Label(self.detection_frame, text="Source:", bootstyle="secondary")
        self.source_label.pack(pady=5)
        self.source_entry = ttk.Entry(self.detection_frame, bootstyle="secondary")
        self.source_entry.insert(0, "default")
        self.source_entry.pack(pady=5)
        
        # Location name parameter
        self.location_label = ttk.Label(self.detection_frame, text="Location Name:", bootstyle="secondary")
        self.location_label.pack(pady=5)
        self.location_entry = ttk.Entry(self.detection_frame, bootstyle="secondary")
        self.location_entry.pack(pady=5)
        
        # Device parameter
        self.device_label = ttk.Label(self.detection_frame, text="Device:", bootstyle="secondary")
        self.device_label.pack(pady=5)
        self.device_entry = ttk.Entry(self.detection_frame, bootstyle="secondary")
        self.device_entry.insert(0, "default")
        self.device_entry.pack(pady=5)

    def start_program_detection(self):
        # Get the parameter values
        source = self.source_entry.get()
        location_name = self.location_entry.get()
        device = self.device_entry.get()
        
        # Assign default values if necessary
        if source == "default":
            source = "0"
        if device == "default":
            device = "cpu"
        
        # Validate the parameters
        if not source:
            messagebox.showerror("Error", "Source parameter not detected.")
            return
        if not device:
            messagebox.showerror("Error", "Device parameter not detected.")
            return
        
        # Start the detection detection program if it is not already running
        if self.process_detection is None:
            if platform.system() == "Windows":
                # Start the terminal command on Windows
                self.process_detection = subprocess.Popen(["cmd", "/c", f"python detect_test_final_TA.py --source {source} --location-name {location_name} --device {device} --conf 0.1 --weights best_v3.pt"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                # Start the terminal command on Unix-based systems
                self.process_detection = subprocess.Popen(["bash", "-c", f"python detect_test_final_TA.py --source {source} --location-name {location_name} --device {device} --conf 0.1 --weights best_v3.pt"], preexec_fn=os.setsid)

    def stop_program_detection(self):
        # Stop the detection detection program if it is running
        if self.process_detection:
            if platform.system() == "Windows":
                # Send CTRL_BREAK_EVENT on Windows to stop the process
                self.process_detection.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Send SIGTERM on Unix-based systems to stop the process
                os.killpg(os.getpgid(self.process_detection.pid), signal.SIGTERM)
            self.process_detection = None  # Reset the process variable

# Create the main window with a specific theme and title
root = ttk.Window(themename="cyborg")
root.title("Detection System GUI")

# Make the window scalable and responsive
window_width = 400
window_height = 300
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))

# Set the window size and position
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
root.resizable(True, True)  # Allow the window to be resizable

# Initialize and run the main application
app = Application(master=root)
app.mainloop()
