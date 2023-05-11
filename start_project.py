import os
import subprocess
import time

current_directory = r"D:\jetbrains projects\pycharm\aim"
# 更新pyenv.cfg
with open(os.path.join(current_directory, "backend", "venv", "pyvenv.cfg"), "w") as cfg_file:
    cfg_file.write(f"home = {os.path.join(current_directory, 'backend', 'venv', 'Python37')}\n")
    cfg_file.write("include-system-site-packages = false\n")
    cfg_file.write("version = 3.7.2\n")


def start_process(command):
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                            shell=True)


def start_redis(current_directory):
    os.chdir(os.path.join(current_directory, "Redis"))
    return start_process("redis-server redis.windows.conf")


def start_celery_worker(current_directory):
    os.chdir(os.path.join(current_directory, "backend"))
    return start_process(".\\venv\\Scripts\\python.exe -m celery -A backend worker -l info -P solo")


def start_django_runserver(current_directory):
    return start_process(".\\venv\\Scripts\\python.exe .\\manage.py runserver 8000")


# main.py

import os
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk


def update_text(widget, process):
    while True and process.poll() is None:
        # time.sleep(1)
        # print(time.time())
        line = process.stdout.readline()
        if line:
            try:
                widget.insert(tk.END, line)
                widget.see(tk.END)
            except RuntimeError as ignore:
                break
        else:
            break
    print("end")

started = False


def start_processes():
    global started
    if started:
        return
    started = True
    print("start processes")
    process1 = start_redis(current_directory)
    t1 = threading.Thread(target=update_text, args=(redis_output, process1))
    t1.start()

    process2 = start_celery_worker(current_directory)
    t2 = threading.Thread(target=update_text, args=(celery_output, process2))
    t2.start()

    process3 = start_django_runserver(current_directory)
    t3 = threading.Thread(target=update_text, args=(django_output, process3))
    t3.start()
    return (process1, process2, process3)


root = tk.Tk()
root.title("Start Project")

main_frame = ttk.Frame(root, padding="3 3 12 12")
main_frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Redis frame
frame1 = ttk.Frame(main_frame, padding="3 3 3 3")
frame1.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)

redis_label = ttk.Label(frame1, text="Redis")
redis_label.grid(column=0, row=0, pady=5, sticky=(tk.W, tk.E))

redis_output = scrolledtext.ScrolledText(frame1, wrap=tk.WORD, width=40, height=20)
redis_output.grid(column=0, row=1, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
frame1.columnconfigure(0, weight=1)
frame1.rowconfigure(1, weight=1)

# Celery worker frame
frame2 = ttk.Frame(main_frame, padding="3 3 3 3")
frame2.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=1)

celery_label = ttk.Label(frame2, text="Celery worker")
celery_label.grid(column=0, row=0, pady=5, sticky=(tk.W, tk.E))

celery_output = scrolledtext.ScrolledText(frame2, wrap=tk.WORD, width=40, height=20)
celery_output.grid(column=0, row=1, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
frame2.columnconfigure(0, weight=1)
frame2.rowconfigure(1, weight=1)

# Django runserver frame
frame3 = ttk.Frame(main_frame, padding="3 3 3 3")
frame3.grid(column=2, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
main_frame.columnconfigure(2, weight=1)
main_frame.rowconfigure(0, weight=1)

django_label = ttk.Label(frame3, text="Django runserver")
django_label.grid(column=0, row=0, pady=5, sticky=(tk.W, tk.E))

django_output = scrolledtext.ScrolledText(frame3, wrap=tk.WORD, width=40, height=20)
django_output.grid(column=0, row=1, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
frame3.columnconfigure(0, weight=1)
frame3.rowconfigure(1, weight=1)

# Start processes when the program starts
processes = start_processes()


def on_closing():
    global stop_threads
    stop_threads = True
    for process in processes:
        process.terminate()
    root.destroy()
    print("root destory")


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
