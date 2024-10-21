import tkinter as tk
from tkinter import filedialog, scrolledtext
from threading import Thread
import time
import os
import sys
from PIL import Image, ImageTk  # Use Pillow to handle .png icons

from DataCollectorProg import DataCollectionV3
# Import the collect_dataset function from your DataCollectionV3 script
from DataCollectorProg.DataCollectionV3 import collect_dataset

# Global variable to control program execution
is_paused = False
is_stopped = False

bg_color = "#2E2E2E"
fg_color = "#FFFFFF"
highlight_color = "#4C4C4C"

# Search highlight variables
search_term = ""
highlighted_indexes = []
current_index = 0


def start_program(path_field, output_window):
    global is_paused, is_stopped
    is_paused = False
    is_stopped = False

    # Update the path variable with the value from the input field
    path = path_field.get()

    if not os.path.exists(path):
        output_window.insert(tk.END, "Invalid path! Please select a valid path.\n")
        return

    # Clear the output window
    output_window.delete(1.0, tk.END)

    # Execute the dataset collection in a separate thread
    thread = Thread(target=run_program, args=(path, output_window))
    thread.start()


def run_program(path, output_window):
    global is_paused, is_stopped

    # Redirect the output to the scrolled text window
    class OutputRedirector:
        def __init__(self, widget):
            self.widget = widget

        def write(self, text):
            self.widget.insert(tk.END, text)
            self.widget.yview(tk.END)  # Auto scroll

        def flush(self):  # Add flush method
            pass

    # Redirect print statements to the output window
    output_redirector = OutputRedirector(output_window)
    sys.stdout = output_redirector

    # Simulating collect_dataset call (replace with actual function)
    while not is_stopped:
        if is_paused:
            output_window.insert(tk.END, "Program paused...\n")
            while is_paused:
                time.sleep(1)
            output_window.insert(tk.END, "Program resumed...\n")

        # Execute the function
        collect_dataset(path)

        if is_stopped:
            output_window.insert(tk.END, "Program stopped.\n")
            break


def pause_program(output_window):
    global is_paused
    is_paused = True
    output_window.insert(tk.END, "Pausing the program...\n")
    DataCollectionV3.pause_event=True


def resume_program(output_window):
    global is_paused
    is_paused = False
    output_window.insert(tk.END, "Resuming the program...\n")
    DataCollectionV3.resume_event = True


def stop_program(output_window):
    global is_stopped
    is_stopped = True
    output_window.insert(tk.END, "Stopping the program...\n")
    DataCollectionV3.running=False
    root.destroy()  # Close the Tkinter window


def browse_directory(path_field):
    directory = filedialog.askdirectory()
    if directory:
        path_field.delete(0, tk.END)
        path_field.insert(0, directory)


def search_text(output_window, search_field, search_counter):
    global search_term, highlighted_indexes, current_index
    search_term = search_field.get()

    # Clear previous highlights
    output_window.tag_remove("highlight", 1.0, tk.END)
    output_window.tag_remove("current", 1.0, tk.END)

    if search_term:
        start_index = "1.0"
        highlighted_indexes = []  # Reset the list of highlighted positions

        # Search for all occurrences of the term and store their indices
        while True:
            start_index = output_window.search(search_term, start_index, nocase=1, stopindex=tk.END)
            if not start_index:
                break
            end_index = f"{start_index}+{len(search_term)}c"
            highlighted_indexes.append((start_index, end_index))
            start_index = end_index  # Move the search forward

        # Reverse the order to highlight last found term first
        highlighted_indexes.reverse()

        # Set the current index to the first match (which was the last occurrence in the original search)
        current_index = 0
        if highlighted_indexes:
            highlight_current(output_window)
            update_search_counter(search_counter)


def highlight_current(output_window):
    global current_index, highlighted_indexes
    if highlighted_indexes:
        output_window.tag_remove("current", 1.0, tk.END)  # Remove previous current highlight

        # Highlight the current search result
        start_index, end_index = highlighted_indexes[current_index]
        output_window.tag_add("current", start_index, end_index)
        output_window.see(start_index)  # Scroll to the current highlighted result
        output_window.tag_configure("current", background="green")  # Green highlight for the current selection


def prev_highlight(output_window, search_counter):
    global current_index, highlighted_indexes
    if highlighted_indexes and current_index < len(highlighted_indexes) - 1:
        current_index += 1
        highlight_current(output_window)
        update_search_counter(search_counter)


def next_highlight(output_window, search_counter):
    global current_index, highlighted_indexes
    if highlighted_indexes and current_index > 0:
        current_index -= 1
        highlight_current(output_window)
        update_search_counter(search_counter)


def update_search_counter(search_counter):
    """Update the counter that shows current search result / total search results."""
    global current_index, highlighted_indexes
    total_results = len(highlighted_indexes)
    current_result = total_results - current_index  # +1 to move from 0-based index to human-readable count
    search_counter.config(text=f"{current_result}/{total_results}")



def main():
    global root  # Declare root as global
    # Create the main window
    root = tk.Tk()
    root.title("Flipkart Grid 6.0 DataSet Collector")
    root.geometry("800x600")

    # Load the custom icon (make sure the path is correct)
    icon_path = "icon.png"  # Replace with your .png or .ico file path
    icon_image = Image.open(icon_path)
    icon_photo = ImageTk.PhotoImage(icon_image)

    # Set the icon for the Tkinter window
    root.iconphoto(False, icon_photo)

    # Create path input and browse button
    path_label = tk.Label(root, text="Path:", )
    path_label.pack(pady=5)

    path_frame = tk.Frame(root, )
    path_frame.pack(pady=5)

    path_field = tk.Entry(path_frame, width=100, justify='center', )
    path_field.pack(side=tk.TOP, fill=tk.X, expand=True, padx=10, pady=5)

    browse_button_size = (85, 2)
    browse_button = tk.Button(root, text="Browse", command=lambda: browse_directory(path_field),
                              width=browse_button_size[0], )
    browse_button.pack(pady=5, padx=50)

    # Create buttons in two columns
    button_frame = tk.Frame(root, )
    button_frame.pack(pady=5)

    # Adjust button size and layout
    button_size = (40, 2)  # Width and height for the buttons
    start_button = tk.Button(button_frame, text="Start", command=lambda: start_program(path_field, output_window),
                             width=button_size[0], )
    start_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    stop_button = tk.Button(button_frame, text="Stop", command=lambda: stop_program(output_window),
                            width=button_size[0], )
    stop_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    pause_button = tk.Button(button_frame, text="Pause", command=lambda: pause_program(output_window),
                             width=button_size[0], )
    pause_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    resume_button = tk.Button(button_frame, text="Resume", command=lambda: resume_program(output_window),
                              width=button_size[0], )
    resume_button.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

    # Create a scrolled text widget to display the output
    output_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20, )
    output_window.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
    output_window.tag_configure("highlight", background="yellow")  # Highlight color

    # Search bar, arrow buttons, and search result counter
    search_frame = tk.Frame(root)
    search_frame.pack(pady=5)

    search_field = tk.Entry(search_frame, width=60)
    search_field.grid(row=0, column=0, padx=10, pady=0)

    search_button = tk.Button(search_frame, text="Search",
                              command=lambda: search_text(output_window, search_field, search_counter))
    search_button.grid(row=0, column=1, padx=5, pady=0)

    up_button = tk.Button(search_frame, text="▲", command=lambda: prev_highlight(output_window, search_counter))
    up_button.grid(row=0, column=2, padx=5, pady=0)

    down_button = tk.Button(search_frame, text="▼", command=lambda: next_highlight(output_window, search_counter))
    down_button.grid(row=0, column=3, padx=5, pady=0)

    # Search result counter
    search_counter = tk.Label(search_frame, text="0/0")
    search_counter.grid(row=0, column=4, padx=5, pady=0)

    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()
