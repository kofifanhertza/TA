import tkinter as tk
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
        self.process_apd = None  # Initialize the APD detection process variable

    def create_widgets(self):
        # Create a style for the Labelframe
        style = ttk.Style()
        style.configure("Custom.TLabelframe.Label", foreground="white")
        
        # APD Detection Section
        # Create a labeled frame for APD Detection with custom style
        self.apd_frame = ttk.Labelframe(self, text="People Detection", bootstyle="primary", style="Custom.TLabelframe")
        self.apd_frame.pack(fill=tk.X, padx=10, pady=5)
        # Create and pack the start button for APD Detection
        self.play_button_apd = ttk.Button(self.apd_frame, text="Start People Detection", command=self.start_program_apd, bootstyle="info", width=20)
        self.play_button_apd.pack(pady=5)
        # Create and pack the stop button for APD Detection
        self.stop_button_apd = ttk.Button(self.apd_frame, text="Stop People Detection", command=self.stop_program_apd, bootstyle="danger", width=20)
        self.stop_button_apd.pack(pady=5)

    def start_program_apd(self):
        # Start the APD detection program if it is not already running
        if self.process_apd is None:
            if platform.system() == "Windows":
                # Start the program on Windows
                self.process_apd = subprocess.Popen(["python", "detect_test_final_TA.py"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                # Start the program on Unix-based systems
                self.process_apd = subprocess.Popen(["python3", "detect_test_final_TA.py"], preexec_fn=os.setsid)

    def stop_program_apd(self):
        # Stop the APD detection program if it is running
        if self.process_apd:
            if platform.system() == "Windows":
                # Send CTRL_BREAK_EVENT on Windows to stop the process
                self.process_apd.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Send SIGTERM on Unix-based systems to stop the process
                os.killpg(os.getpgid(self.process_apd.pid), signal.SIGTERM)
            self.process_apd = None  # Reset the process variable

# Create the main window with a specific theme and title
root = ttk.Window(themename="superhero")
root.title("Detection System GUI")

# Make the window scalable and responsive
window_width = 400
window_height = 200
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