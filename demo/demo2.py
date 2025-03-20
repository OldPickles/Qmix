import tkinter as tk

try:
    root = tk.Tk()
    print("Tkinter is installed and configured correctly.")
except Exception as e:
    print(f"Tkinter not found or misconfigured: {e}")