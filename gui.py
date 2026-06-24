import sys
import os
import json
import threading
import time
import subprocess
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
import customtkinter as ctk
from tkinter import messagebox, filedialog
import traceback
import re

# Force Permanent Light Mode
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

import platform

# Detect system platform for typography fallbacks
sys_name = platform.system()
if sys_name == "Windows":
    FONT_FAMILY = "Segoe UI"
elif sys_name == "Darwin":
    FONT_FAMILY = ".AppleSystemUIFont"
else:
    FONT_FAMILY = "Ubuntu"  # Linux fallback

FONT_TITLE = (FONT_FAMILY, 28, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 13)
FONT_HEADING = (FONT_FAMILY, 20, "bold")
FONT_CARD_TITLE = (FONT_FAMILY, 15, "bold")
FONT_CARD_DESC = (FONT_FAMILY, 11)
FONT_LABEL = (FONT_FAMILY, 11, "bold")
FONT_BUTTON = (FONT_FAMILY, 12, "bold")

DB_FILE = "database.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        model TEXT DEFAULT 'pix2pix',
        direction TEXT DEFAULT 'AtoB',
        netG TEXT DEFAULT 'unet_256',
        dataset_mode TEXT DEFAULT 'aligned',
        norm TEXT DEFAULT 'batch',
        batch_size INTEGER DEFAULT 4,
        n_epochs INTEGER DEFAULT 100,
        n_epochs_decay INTEGER DEFAULT 100,
        dataroot TEXT DEFAULT './datasets/guwahati_azara_processed',
        gpu_ids TEXT DEFAULT '0',
        use_wandb INTEGER DEFAULT 0,
        epoch_count INTEGER,
        status TEXT DEFAULT 'pending',
        pid INTEGER,
        created_at TEXT,
        completed_at TEXT,
        log_file TEXT,
        is_finetuning INTEGER DEFAULT 0,
        parent_run_id INTEGER,
        parent_epoch TEXT
    )
    """)
    conn.commit()
    conn.close()

def db_get_all_runs():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM training_runs ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_run(run_id):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def db_create_run(data):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if "created_at" not in data:
        data["created_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    keys = list(data.keys())
    values = list(data.values())
    placeholders = ", ".join(["?"] * len(keys))
    fields = ", ".join(keys)
    cursor.execute(f"INSERT INTO training_runs ({fields}) VALUES ({placeholders})", values)
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id

def db_update_run(run_id, updates):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
    values = list(updates.values()) + [run_id]
    cursor.execute(f"UPDATE training_runs SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()

class PatternBackground(ctk.CTkCanvas):
    """Draws a professional dotted grid pattern in the app background."""
    def __init__(self, master, **kwargs):
        super().__init__(master, bg="#fcfdfe", highlightthickness=0, **kwargs)
        self.bind("<Configure>", self.draw_pattern)
        
    def draw_pattern(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        
        # Draw a beautiful soft dot grid
        dot_distance = 30
        for x in range(15, w, dot_distance):
            for y in range(15, h, dot_distance):
                self.create_oval(x, y, x+2, y+2, fill="#e5e8eb", outline="")


class Pix2PixGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pix2Pix Control Center")
        self.geometry("1180x920")
        self.resizable(True, True)
        self.configure(fg_color="#fcfdfe")
        
        # Log streaming state
        self.processes = {}  # run_id -> subprocess.Popen
        self.active_run_id = None
        
        # Auto-refresh threads
        self.refresh_runs_active = True
        self.refresh_gpu_active = True
        
        # Base container layout
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True)
        
        # Instantiate pages
        self.home_page = HomePage(self.container, self)
        self.train_page = TrainPage(self.container, self)
        self.finetune_page = FinetunePage(self.container, self)
        self.runs_page = RunsPage(self.container, self)
        self.checkpoints_page = CheckpointsPage(self.container, self)
        self.preprocess_page = PreprocessPage(self.container, self)
        self.test_model_page = TestModelPage(self.container, self)
        
        # Show home page initially
        self.show_page("home")
        
        # Start background update tasks
        threading.Thread(target=self.gpu_monitoring_loop, daemon=True).start()
        threading.Thread(target=self.runs_monitoring_loop, daemon=True).start()

    def show_page(self, page_name: str):
        self.home_page.pack_forget()
        self.train_page.pack_forget()
        self.finetune_page.pack_forget()
        self.runs_page.pack_forget()
        self.checkpoints_page.pack_forget()
        self.preprocess_page.pack_forget()
        self.test_model_page.pack_forget()
        
        if page_name == "home":
            self.home_page.pack(fill="both", expand=True)
            self.home_page.on_show()
        elif page_name == "train":
            self.train_page.pack(fill="both", expand=True, padx=30, pady=30)
            self.train_page.on_show()
        elif page_name == "finetune":
            self.finetune_page.pack(fill="both", expand=True, padx=30, pady=30)
            self.finetune_page.on_show()
        elif page_name == "runs":
            self.runs_page.pack(fill="both", expand=True, padx=30, pady=30)
            self.runs_page.on_show()
        elif page_name == "checkpoints":
            self.checkpoints_page.pack(fill="both", expand=True, padx=30, pady=30)
            self.checkpoints_page.on_show()
        elif page_name == "preprocess":
            self.preprocess_page.pack(fill="both", expand=True, padx=30, pady=30)
            self.preprocess_page.on_show()
        elif page_name == "test_model":
            self.test_model_page.pack(fill="both", expand=True, padx=30, pady=30)
            self.test_model_page.on_show()



    def get_gpu_info_direct(self):
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1.0
            )
            if res.returncode == 0:
                lines = res.stdout.strip().split("\n")
                gpus = []
                for line in lines:
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    gpus.append({
                        "utilization_gpu_percent": float(parts[3]),
                        "memory_used_mib": float(parts[6]),
                        "memory_total_mib": float(parts[5]),
                        "temperature_c": float(parts[2])
                    })
                return gpus
        except Exception:
            pass
        return []

    def gpu_monitoring_loop(self):
        while self.refresh_gpu_active:
            gpus = self.get_gpu_info_direct()
            if gpus:
                gpu = gpus[0]
                self.after(0, lambda u=gpu["utilization_gpu_percent"], mu=gpu["memory_used_mib"], mt=gpu["memory_total_mib"], t=gpu["temperature_c"]: self.home_page.update_gpu(u, mu, mt, t))
            else:
                self.after(0, lambda: self.home_page.update_gpu(None, None, None, None))
            time.sleep(3.0)

    def runs_monitoring_loop(self):
        while self.refresh_runs_active:
            try:
                runs = db_get_all_runs()
                self.after(0, lambda r_data=runs: self.runs_page.update_runs_list(r_data))
            except Exception as e:
                print(f"Error in runs monitoring loop: {e}")
            time.sleep(4.0)

    def start_training_direct(self, run_id):
        run = db_get_run(run_id)
        if not run:
            return
            
        cmd = [
            sys.executable, "train.py",
            "--dataroot", run.get("dataroot") if run.get("dataroot") else "./datasets/guwahati_azara_processed",
            "--name", run.get("name"),
            "--model", run.get("model"),
            "--direction", run.get("direction"),
            "--netG", run.get("netG"),
            "--dataset_mode", run.get("dataset_mode"),
            "--norm", run.get("norm"),
            "--pool_size", "0",
            "--batch_size", str(run.get("batch_size")),
            "--n_epochs", str(run.get("n_epochs")),
            "--n_epochs_decay", str(run.get("n_epochs_decay"))
        ]
        if run.get("use_wandb"):
            cmd.append("--use_wandb")
        if run.get("is_finetuning"):
            cmd.extend([
                "--continue_train",
                "--epoch", str(run.get("parent_epoch") or "latest")
            ])
            if run.get("epoch_count"):
                cmd.extend(["--epoch_count", str(run.get("epoch_count"))])
                
        # Setup log file
        log_dir = Path("./logs") / run.get("name")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"run_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Save log file path in DB
        db_update_run(run_id, {
            "status": "running",
            "log_file": str(log_file_path)
        })
        
        env = os.environ.copy()
        gpu_ids = run.get("gpu_ids", "0")
        if gpu_ids:
            if gpu_ids.lower() == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids
                
        # Start subprocess dynamically based on platform (Windows vs Unix)
        kwargs = {}
        if platform.system() != "Windows":
            kwargs["preexec_fn"] = os.setsid
            
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).parent.absolute()),
            **kwargs
        )
        
        db_update_run(run_id, {
            "pid": process.pid
        })
        
        self.processes[run_id] = process
        
        # Start log parsing and file writer thread
        threading.Thread(
            target=self._monitor_process_stdout,
            args=(run_id, process, log_file_path),
            daemon=True
        ).start()

    def _monitor_process_stdout(self, run_id, process, log_file_path):
        try:
            with open(log_file_path, "w", encoding="utf-8") as log_file:
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        decoded_line = line.decode("utf-8", errors="ignore")
                        # Write to file
                        log_file.write(decoded_line)
                        log_file.flush()
                        
                        # Send updates to GUI if this is the active run
                        self.after(0, lambda msg=decoded_line: self.on_direct_log_message(run_id, msg))
            
            # Process exited
            return_code = process.poll()
            status = "completed" if return_code == 0 else "failed"
            current_run = db_get_run(run_id)
            if current_run and current_run.get("status") not in ["stopped", "failed"]:
                db_update_run(run_id, {
                    "status": status,
                    "completed_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                    "pid": None
                })
        except Exception as e:
            print(f"Error in monitor process thread: {e}")
            db_update_run(run_id, {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "pid": None
            })
        finally:
            if run_id in self.processes:
                del self.processes[run_id]
            self.refresh_runs_list_directly()

    def on_direct_log_message(self, run_id, message):
        if self.active_run_id == run_id:
            self.runs_page.handle_direct_log_line(message)

    def refresh_runs_list_directly(self):
        try:
            runs = db_get_all_runs()
            self.runs_page.update_runs_list(runs)
        except Exception:
            pass

    def stop_training_direct(self, run_id) -> bool:
        process = self.processes.get(run_id)
        if not process:
            run = db_get_run(run_id)
            if run and run.get("status") == "running":
                db_update_run(run_id, {
                    "status": "stopped",
                    "pid": None,
                    "completed_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
                })
                self.refresh_runs_list_directly()
                return True
            return False
            
        try:
            if platform.system() == "Windows":
                # Taskkill forces termination of process tree
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
            else:
                import signal
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            db_update_run(run_id, {
                "status": "stopped",
                "pid": None,
                "completed_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
            })
            if run_id in self.processes:
                del self.processes[run_id]
            self.refresh_runs_list_directly()
            return True
        except Exception as e:
            print(f"Error terminating process: {e}")
            return False

    def on_closing(self):
        self.refresh_runs_active = False
        self.refresh_gpu_active = False
        for r_id, proc in list(self.processes.items()):
            try:
                if proc.poll() is None:
                    if platform.system() == "Windows":
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                    else:
                        import signal
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
        try:
            if hasattr(self, "preprocess_page") and self.preprocess_page.process:
                if self.preprocess_page.process.poll() is None:
                    if platform.system() == "Windows":
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.preprocess_page.process.pid)], capture_output=True)
                    else:
                        import signal
                        os.killpg(os.getpgid(self.preprocess_page.process.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            if hasattr(self, "test_model_page") and hasattr(self.test_model_page, "netG"):
                self.test_model_page.netG = None
        except Exception:
            pass
        self.destroy()
        sys.exit(0)


# ==========================================
# MODAL DIALOGS FOR TRAINING SETTINGS
# ==========================================
class TrainSettingsModal(ctk.CTkToplevel):
    def __init__(self, parent_page):
        # CustomTkinter CTkToplevel requires master to be a CTk window (controller)
        super().__init__(master=parent_page.controller)
        self.parent = parent_page
        
        self.title("Configure Training Parameters")
        self.geometry("450x540")
        self.resizable(False, False)
        self.configure(fg_color="#f8fafc")
        
        # Route close window to clean destroy
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        
        # Lift and lock focus safely
        self.transient(parent_page.controller)
        self.focus_force()
        
        container = ctk.CTkFrame(self, fg_color="#ffffff", corner_radius=12, border_width=1, border_color="#e2e8f0")
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(container, text="⚙️ Training Parameters", font=(FONT_FAMILY, 15, "bold"), text_color="#1a73e8").pack(anchor="w", padx=25, pady=(15, 10))
        
        # Model Strategy
        ctk.CTkLabel(container, text="Model Strategy:", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=25, pady=(5, 2))
        self.model_opt = ctk.CTkOptionMenu(container, width=320, values=["pix2pix"])
        self.model_opt.pack(padx=25, pady=2)
        self.model_opt.set(self.parent.model_strategy)

        # Batch Size Option
        ctk.CTkLabel(container, text="Batch Size:", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=25, pady=(5, 2))
        self.batch_opt = ctk.CTkOptionMenu(container, width=320, values=["4", "1", "2", "8", "16"])
        self.batch_opt.pack(padx=25, pady=2)
        self.batch_opt.set(self.parent.batch_size)
        
        # Constant Epochs
        ctk.CTkLabel(container, text="Constant Learning Rate Epochs (n_epochs):", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=25, pady=(5, 2))
        self.epochs_opt = ctk.CTkOptionMenu(container, width=320, values=["25", "10", "50", "100", "200"])
        self.epochs_opt.pack(padx=25, pady=2)
        self.epochs_opt.set(self.parent.n_epochs)

        # Decay Epochs
        ctk.CTkLabel(container, text="Decaying Learning Rate Epochs (n_epochs_decay):", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=25, pady=(5, 2))
        self.decay_opt = ctk.CTkOptionMenu(container, width=320, values=["25", "0", "10", "50", "100", "200"])
        self.decay_opt.pack(padx=25, pady=2)
        self.decay_opt.set(self.parent.n_epochs_decay)

        # GPU Device IDs
        ctk.CTkLabel(container, text="GPU Core Selector:", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=25, pady=(5, 2))
        self.gpu_opt = ctk.CTkOptionMenu(container, width=320, values=["0", "1", "cpu"])
        self.gpu_opt.pack(padx=25, pady=2)
        self.gpu_opt.set(self.parent.gpu_ids)

        # WandB Tracker Checkbox
        self.wandb_check = ctk.CTkCheckBox(container, text="Track metrics with Weights & Biases (WandB)", font=FONT_LABEL, text_color="#334155")
        self.wandb_check.pack(anchor="w", padx=25, pady=12)
        if self.parent.use_wandb:
            self.wandb_check.select()
        else:
            self.wandb_check.deselect()

        # Save Button
        save_btn = ctk.CTkButton(
            container, 
            text="Save Settings", 
            font=FONT_BUTTON, 
            height=36,
            fg_color="#1a73e8",
            text_color="#ffffff",
            hover_color="#155cb4",
            command=self.save_settings
        )
        save_btn.pack(fill="x", padx=25, pady=(15, 5))

        # Close / Cancel Button
        close_btn = ctk.CTkButton(
            container, 
            text="Cancel & Close", 
            font=FONT_BUTTON, 
            height=36,
            fg_color="#f1f5f9",
            text_color="#0f172a",
            hover_color="#e2e8f0",
            border_width=1,
            border_color="#cbd5e1",
            command=self.destroy
        )
        close_btn.pack(fill="x", padx=25, pady=(5, 15))

    def save_settings(self):
        try:
            self.parent.model_strategy = self.model_opt.get()
            self.parent.batch_size = self.batch_opt.get()
            self.parent.n_epochs = self.epochs_opt.get()
            self.parent.n_epochs_decay = self.decay_opt.get()
            self.parent.gpu_ids = self.gpu_opt.get()
            self.parent.use_wandb = self.wandb_check.get() == 1
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
        finally:
            self.destroy()

    def destroy(self):
        self.parent.settings_btn.configure(state="normal")
        super().destroy()


# ==========================================
# MODAL DIALOGS FOR FINETUNING SETTINGS
# ==========================================
class FineSettingsModal(ctk.CTkToplevel):
    def __init__(self, parent_page):
        # CustomTkinter CTkToplevel requires master to be a CTk window (controller)
        super().__init__(master=parent_page.controller)
        self.parent = parent_page
        
        self.title("Configure Fine-Tuning Parameters")
        self.geometry("450x530")
        self.resizable(False, False)
        self.configure(fg_color="#f8fafc")
        
        # Route close window to clean destroy
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        
        # Lift and lock focus safely
        self.transient(parent_page.controller)
        self.focus_force()
        
        container = ctk.CTkFrame(self, fg_color="#ffffff", corner_radius=12, border_width=1, border_color="#e2e8f0")
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(container, text="⚙️ Fine-Tuning Parameters", font=(FONT_FAMILY, 15, "bold"), text_color="#0f9d58").pack(anchor="w", padx=25, pady=(15, 10))
        
        # Epoch checkpoint to load
        ctk.CTkLabel(container, text="Checkpoint Suffix (parent_epoch):", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=25, pady=(5, 2))
        self.epoch_opt = ctk.CTkOptionMenu(container, width=320, values=self.parent.checkpoint_epochs, command=self.on_epoch_changed)
        self.epoch_opt.pack(padx=25, pady=2)
        self.epoch_opt.set(self.parent.fine_epoch)

        # Starting count
        ctk.CTkLabel(container, text="Start Counter index (epoch_count):", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=25, pady=(5, 2))
        self.start_count_opt = ctk.CTkOptionMenu(container, width=320, values=["51", "1", "26", "101"])
        self.start_count_opt.pack(padx=25, pady=2)
        self.start_count_opt.set(self.parent.fine_start_count)

        # Batch size
        ctk.CTkLabel(container, text="Batch Size:", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=25, pady=(5, 2))
        self.batch_opt = ctk.CTkOptionMenu(container, width=320, values=["4", "1", "2", "8", "16"])
        self.batch_opt.pack(padx=25, pady=2)
        self.batch_opt.set(self.parent.fine_batch)

        # Constant fine-tuning epochs
        ctk.CTkLabel(container, text="Fine-Tuning Constant Epochs (n_epochs):", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=25, pady=(5, 2))
        self.epochs_opt = ctk.CTkOptionMenu(container, width=320, values=["25", "5", "10", "50", "100"])
        self.epochs_opt.pack(padx=25, pady=2)
        self.epochs_opt.set(self.parent.fine_epochs)

        # Decay fine-tuning epochs
        ctk.CTkLabel(container, text="Fine-Tuning Decaying Epochs (n_epochs_decay):", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=25, pady=(5, 2))
        self.decay_opt = ctk.CTkOptionMenu(container, width=320, values=["25", "0", "5", "10", "50", "100"])
        self.decay_opt.pack(padx=25, pady=2)
        self.decay_opt.set(self.parent.fine_epochs_decay)

        # Save Button
        save_btn = ctk.CTkButton(
            container, 
            text="Save Settings", 
            font=FONT_BUTTON, 
            height=36,
            fg_color="#0f9d58",
            text_color="#ffffff",
            hover_color="#0b7a44",
            command=self.save_settings
        )
        save_btn.pack(fill="x", padx=25, pady=(20, 5))

        # Close / Cancel Button
        close_btn = ctk.CTkButton(
            container, 
            text="Cancel & Close", 
            font=FONT_BUTTON, 
            height=36,
            fg_color="#f1f5f9",
            text_color="#0f172a",
            hover_color="#e2e8f0",
            border_width=1,
            border_color="#cbd5e1",
            command=self.destroy
        )
        close_btn.pack(fill="x", padx=25, pady=(5, 15))

    def on_epoch_changed(self, epoch: str):
        if epoch.isdigit():
            nxt = int(epoch) + 1
            self.start_count_opt.configure(values=[str(nxt), "1", "26", "51", "101"])
            self.start_count_opt.set(str(nxt))
        else:
            self.start_count_opt.set("51")

    def save_settings(self):
        try:
            self.parent.fine_epoch = self.epoch_opt.get()
            self.parent.fine_start_count = self.start_count_opt.get()
            self.parent.fine_batch = self.batch_opt.get()
            self.parent.fine_epochs = self.epochs_opt.get()
            self.parent.fine_epochs_decay = self.decay_opt.get()
            self.parent.update_direction_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
        finally:
            self.destroy()

    def destroy(self):
        self.parent.settings_btn.configure(state="normal")
        super().destroy()


# ==========================================
# PAGE 1: DASHBOARD
# ==========================================
class HomePage(ctk.CTkFrame):
    def __init__(self, parent, controller: Pix2PixGUI):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        
        self.bg_pattern = PatternBackground(self)
        self.bg_pattern.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.fg_wrapper = ctk.CTkFrame(self, fg_color="transparent")
        self.fg_wrapper.pack(fill="both", expand=True, padx=30, pady=30)

        # Header Block
        self.header_frame = ctk.CTkFrame(self.fg_wrapper, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(5, 15))
        
        self.badge = ctk.CTkFrame(self.header_frame, fg_color="#e2eafd", corner_radius=12, height=24)
        self.badge.pack(anchor="w")
        self.badge_lbl = ctk.CTkLabel(self.badge, text=" PLATFORM WORKSPACE ", font=(FONT_FAMILY, 9, "bold"), text_color="#1a73e8")
        self.badge_lbl.pack(padx=8, pady=1)

        self.title_label = ctk.CTkLabel(self.header_frame, text="Pix2Pix Dashboard", font=FONT_TITLE, text_color="#1e293b")
        self.title_label.pack(anchor="w", pady=(8, 2))
        self.subtitle_label = ctk.CTkLabel(self.header_frame, text="Select an action to launch, configure, or review your model parameters.", font=FONT_SUBTITLE, text_color="#64748b")
        self.subtitle_label.pack(anchor="w")
        
        # Status Bar
        self.status_bar = ctk.CTkFrame(self.fg_wrapper, fg_color="#ffffff", corner_radius=12, border_width=1, border_color="#e2e8f0")
        self.status_bar.pack(fill="x", pady=(0, 20), ipady=6)
        
        self.status_indicator = ctk.CTkFrame(self.status_bar, width=10, height=10, corner_radius=5, fg_color="#10b981")
        self.status_indicator.pack(side="left", padx=(20, 5))
        
        self.status_lbl = ctk.CTkLabel(self.status_bar, text="SYSTEM STATUS: DIRECT MODE ACTIVE", font=(FONT_FAMILY, 11, "bold"), text_color="#10b981")
        self.status_lbl.pack(side="left", padx=5)
        
        self.gpu_lbl = ctk.CTkLabel(self.status_bar, text="GPU Stats: Offline", font=(FONT_FAMILY, 11, "bold"), text_color="#475569")
        self.gpu_lbl.pack(side="right", padx=20)
        
        # Cards Grid Frame - 2 rows, 3 columns to optimize space
        self.grid_frame = ctk.CTkFrame(self.fg_wrapper, fg_color="transparent")
        self.grid_frame.pack(fill="both", expand=True)
        self.grid_frame.grid_columnconfigure(0, weight=1)
        self.grid_frame.grid_columnconfigure(1, weight=1)
        self.grid_frame.grid_columnconfigure(2, weight=1)
        
        # Create Cards
        self.create_card(0, 0, "Train New Model", "Configure variables and start a new generative model training task.", lambda: self.controller.show_page("train"), "#1a73e8", "#e8f0fe")
        self.create_card(0, 1, "Fine-tune Checkpoint", "Choose a pre-trained epoch checkpoint and configure parameters to resume.", lambda: self.controller.show_page("finetune"), "#0f9d58", "#e6f4ea")
        self.create_card(0, 2, "Monitoring & Real-time Logs", "Watch terminal standard output stream and monitor loss statistics live.", lambda: self.controller.show_page("runs"), "#8b5cf6", "#f5f3ff")
        self.create_card(1, 0, "Model Storage Explorer", "Browse checkpoints folder, review weight files, and clean up workspace storage.", lambda: self.controller.show_page("checkpoints"), "#f97316", "#fff7ed")
        self.create_card(1, 1, "Dataset Preprocessing", "Select raw Landsat satellite scenes and preprocess/tile them for training.", lambda: messagebox.showerror("Platform Not Supported", "Dataset Preprocessing is only supported on Linux and Windows due to specialized satellite science and remote sensing library dependencies.") if platform.system() not in ("Linux", "Windows") else self.controller.show_page("preprocess"), "#0284c7", "#f0f9ff")
        self.create_card(1, 2, "Interactive Model Tester", "Load checkpoints, select test images, and run immediate interactive inference.", lambda: self.controller.show_page("test_model"), "#0d9488", "#f0fdfa")

    def create_card(self, row: int, col: int, title: str, desc: str, command, accent_color: str, hover_bg: str):
        card = ctk.CTkFrame(self.grid_frame, width=320, height=200, fg_color="#ffffff", corner_radius=16, border_width=1.5, border_color="#e2e8f0")
        card.grid(row=row, column=col, padx=15, pady=15)
        card.pack_propagate(False)
        
        def on_enter(e):
            card.configure(border_color=accent_color, fg_color=hover_bg)
            
        def on_leave(e):
            card.configure(border_color="#e2e8f0", fg_color="#ffffff")
            
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        
        top_bar = ctk.CTkFrame(card, height=4, corner_radius=0, fg_color=accent_color)
        top_bar.pack(fill="x", side="top")
        
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=(15, 18))
        
        inner.bind("<Enter>", on_enter)
        inner.bind("<Leave>", on_leave)
            
        lbl_title = ctk.CTkLabel(inner, text=title, font=FONT_CARD_TITLE, text_color="#0f172a")
        lbl_title.pack(anchor="w")
        lbl_title.bind("<Enter>", on_enter)
        lbl_title.bind("<Leave>", on_leave)
        
        lbl_desc = ctk.CTkLabel(inner, text=desc, font=FONT_CARD_DESC, text_color="#64748b", wraplength=275, justify="left")
        lbl_desc.pack(anchor="w", pady=(6, 12), fill="both", expand=True)
        lbl_desc.bind("<Enter>", on_enter)
        lbl_desc.bind("<Leave>", on_leave)
        
        btn = ctk.CTkButton(
            inner, 
            text="Open Action ➔", 
            font=FONT_BUTTON, 
            height=36, 
            corner_radius=18, 
            fg_color=accent_color,
            hover_color=accent_color,
            text_color="#ffffff",
            command=command
        )
        btn.pack(fill="x", side="bottom")



    def update_gpu(self, util, mem_used, mem_total, temp):
        if util is not None:
            self.gpu_lbl.configure(
                text=f"GPU Core: {util}%  |  Temp: {temp}°C  |  VRAM: {int(mem_used)}/{int(mem_total)} MiB"
            )
        else:
            self.gpu_lbl.configure(text="GPU Monitor: Offline")

    def on_show(self):
        pass


# ==========================================
# PAGE 2: TRAIN MODEL PAGE
# ==========================================
class TrainPage(ctk.CTkFrame):
    def __init__(self, parent, controller: Pix2PixGUI):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller

        # Hyperparameter states stored directly in instance attributes to avoid form clutter
        self.batch_size = "4"
        self.n_epochs = "25"
        self.n_epochs_decay = "25"
        self.gpu_ids = "0"
        self.use_wandb = False
        self.model_strategy = "pix2pix"
        self.direction = "AtoB"
        
        # Image Cache to prevent garbage collection
        self.cached_left_img = None
        self.cached_right_img = None

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        back_btn = ctk.CTkButton(
            header, 
            text="⏴ Dashboard", 
            font=FONT_BUTTON, 
            width=130, 
            height=34, 
            corner_radius=17, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1,
            border_color="#cbd5e1",
            command=lambda: self.controller.show_page("home")
        )
        back_btn.pack(side="left")
        
        title = ctk.CTkLabel(header, text="  Train Generative Model", font=FONT_HEADING, text_color="#0f172a")
        title.pack(side="left", padx=10)
        
        # Modern Settings Gear Icon Button (Textless)
        self.settings_btn = ctk.CTkButton(
            header, 
            text="⚙", 
            font=(FONT_FAMILY, 20, "bold"), 
            width=36, 
            height=36, 
            corner_radius=18, 
            fg_color="#f8fafc", 
            text_color="#1a73e8", 
            hover_color="#edf2f7", 
            border_width=1.5,
            border_color="#1a73e8",
            command=self.open_settings_modal
        )
        self.settings_btn.pack(side="right")

        # Form panel
        form = ctk.CTkFrame(self, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#e2e8f0")
        form.pack(fill="both", expand=True, ipady=15)
        
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)

        # Name Entry
        ctk.CTkLabel(form, text="Experiment Run Name:", font=FONT_LABEL, text_color="#334155").grid(row=0, column=0, padx=30, pady=(20, 2), sticky="w")
        self.inp_name = ctk.CTkEntry(form, width=320, height=35, corner_radius=8, font=(FONT_FAMILY, 12))
        self.inp_name.grid(row=1, column=0, padx=30, pady=2, sticky="w")
        self.inp_name.insert(0, "guwahati_azara_pix2pix")

        # Dataset dropdown + browse
        ctk.CTkLabel(form, text="Select Dataroot Directory:", font=FONT_LABEL, text_color="#334155").grid(row=0, column=1, padx=30, pady=(20, 2), sticky="w")
        dataset_row = ctk.CTkFrame(form, fg_color="transparent")
        dataset_row.grid(row=1, column=1, padx=30, pady=2, sticky="ew")
        
        self.dataset_dropdown = ctk.CTkOptionMenu(dataset_row, width=220, height=35, corner_radius=8, values=["./datasets/guwahati_azara_processed"], command=self.update_direction_preview)
        self.dataset_dropdown.pack(side="left")
        
        browse_btn = ctk.CTkButton(
            dataset_row, 
            text="Browse...", 
            font=FONT_BUTTON, 
            width=90, 
            height=35, 
            corner_radius=8, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1, 
            border_color="#cbd5e1",
            command=self.browse_dataset_dir
        )
        browse_btn.pack(side="left", padx=5)

        # Translation preview container
        ctk.CTkLabel(form, text="Image Translation Direction Preview:", font=FONT_LABEL, text_color="#334155").grid(row=2, column=0, columnspan=2, padx=30, pady=(25, 2), sticky="w")
        
        self.preview_container = ctk.CTkFrame(form, fg_color="#f8fafc", corner_radius=12, border_width=1, border_color="#e2e8f0", height=185)
        self.preview_container.grid(row=3, column=0, columnspan=2, padx=30, pady=5, sticky="ew")
        self.preview_container.grid_propagate(False)
        
        self.img_lbl_left = ctk.CTkLabel(self.preview_container, text="No Image", font=(FONT_FAMILY, 10, "bold"), fg_color="#e2e8f0", width=145, height=145, corner_radius=8)
        self.img_lbl_left.pack(side="left", padx=(35, 10), pady=18)
        
        self.arrow_btn = ctk.CTkButton(
            self.preview_container, 
            text="->", 
            font=(FONT_FAMILY, 24, "bold"), 
            text_color="#1a73e8", 
            fg_color="transparent",
            hover_color="#e2e8f0",
            width=50,
            height=50,
            corner_radius=25,
            command=self.toggle_direction
        )
        self.arrow_btn.pack(side="left", padx=20, pady=18)
        
        self.img_lbl_right = ctk.CTkLabel(self.preview_container, text="No Image", font=(FONT_FAMILY, 10, "bold"), fg_color="#e2e8f0", width=145, height=145, corner_radius=8)
        self.img_lbl_right.pack(side="left", padx=(10, 35), pady=18)
        
        self.preview_info_lbl = ctk.CTkLabel(self.preview_container, text="", font=(FONT_FAMILY, 12, "bold"), text_color="#334155")
        self.preview_info_lbl.pack(side="left", padx=20, pady=18)

        # Launch Button
        self.start_btn = ctk.CTkButton(
            self, 
            text="⚡ Launch Training Subprocess", 
            font=(FONT_FAMILY, 13, "bold"), 
            height=46, 
            corner_radius=23, 
            fg_color="#1a73e8", 
            hover_color="#155cb4",
            text_color="#ffffff",
            command=self.start_training
        )
        self.start_btn.pack(fill="x", pady=(20, 0))

    def open_settings_modal(self):
        try:
            self.settings_btn.configure(state="disabled")
            TrainSettingsModal(self)
        except Exception as e:
            self.settings_btn.configure(state="normal")
            messagebox.showerror("Error", f"Failed to open settings dialog: {e}")
            traceback.print_exc()

    def load_dataset_sample(self):
        try:
            dataroot = self.dataset_dropdown.get()
            train_dir = Path(dataroot) / "train"
            if not train_dir.exists():
                return None
            
            # Find first PNG
            pngs = list(train_dir.glob("*.png"))
            if not pngs:
                return None
                
            img = Image.open(pngs[0])
            w, h = img.size
            w2 = int(w / 2)
            
            # Crop A (left) and B (right)
            img_A = img.crop((0, 0, w2, h))
            img_B = img.crop((w2, 0, w, h))
            
            return img_A, img_B
        except Exception as e:
            print(f"Error loading dataset sample: {e}")
            return None

    def toggle_direction(self):
            if self.direction == "AtoB":
                self.direction = "BtoA"
            else:
                self.direction = "AtoB"
            self.update_direction_preview()

    def update_direction_preview(self, choice=None):
        direction = self.direction
        sample = self.load_dataset_sample()
        
        if sample:
            img_A, img_B = sample
            if direction == "AtoB":
                left_img = img_A
                right_img = img_B
                info_text = "Input: Thermal IR  ->  Output: Visible RGB"
                self.arrow_btn.configure(text="->")
            else:
                left_img = img_B
                right_img = img_A
                info_text = "Input: Visible RGB  <-  Output: Thermal IR"
                self.arrow_btn.configure(text="<-")
                
            self.cached_left_img = ctk.CTkImage(light_image=left_img, size=(135, 135))
            self.cached_right_img = ctk.CTkImage(light_image=right_img, size=(135, 135))
            
            # Configure and keep references attached to widgets to prevent GC
            self.img_lbl_left.configure(image=self.cached_left_img, text="")
            self.img_lbl_left.image = self.cached_left_img
            self.img_lbl_right.configure(image=self.cached_right_img, text="")
            self.img_lbl_right.image = self.cached_right_img
            self.preview_info_lbl.configure(text=info_text)
        else:
            self.cached_left_img = None
            self.cached_right_img = None
            self.img_lbl_left.image = None
            self.img_lbl_right.image = None
            
            if direction == "AtoB":
                self.img_lbl_left.configure(image=None, text="THERMAL IR", fg_color="#e2eafd")
                self.img_lbl_right.configure(image=None, text="VISIBLE RGB", fg_color="#e8f8f0")
                self.preview_info_lbl.configure(text="Input: Thermal IR  ->  Output: Visible RGB")
                self.arrow_btn.configure(text="->")
            else:
                self.img_lbl_left.configure(image=None, text="VISIBLE RGB", fg_color="#e8f8f0")
                self.img_lbl_right.configure(image=None, text="THERMAL IR", fg_color="#e2eafd")
                self.preview_info_lbl.configure(text="Input: Visible RGB  <-  Output: Thermal IR")
                self.arrow_btn.configure(text="<-")

    def browse_dataset_dir(self):
        directory = filedialog.askdirectory(initialdir="./datasets")
        if directory:
            try:
                rel = Path(directory).relative_to(Path.cwd())
                path_str = f"./{rel}"
            except ValueError:
                path_str = directory
            
            current_vals = self.dataset_dropdown.cget("values")
            if path_str not in current_vals:
                new_vals = list(current_vals) + [path_str]
                self.dataset_dropdown.configure(values=new_vals)
            self.dataset_dropdown.set(path_str)
            self.update_direction_preview()

    def start_training(self):
        payload = {
            "name": self.inp_name.get().strip(),
            "dataroot": self.dataset_dropdown.get(),
            "model": self.model_strategy,
            "direction": self.direction,
            "netG": "unet_256",
            "dataset_mode": "aligned",
            "norm": "batch",
            "batch_size": int(self.batch_size),
            "n_epochs": int(self.n_epochs),
            "n_epochs_decay": int(self.n_epochs_decay),
            "gpu_ids": self.gpu_ids,
            "use_wandb": int(self.use_wandb),
            "is_finetuning": 0,
            "status": "pending"
        }

        if not payload["name"]:
            messagebox.showerror("Error", "Please fill in the Experiment Run Name.")
            return

        try:
            run_id = db_create_run(payload)
            self.controller.start_training_direct(run_id)
            self.controller.show_page("runs")
            self.controller.runs_page.select_run_directly(run_id)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_show(self):
        datasets_path = Path("./datasets")
        subdirs = ["./datasets/guwahati_azara_processed"]
        if datasets_path.exists():
            for d in datasets_path.iterdir():
                if d.is_dir() and d.name != "bibtex" and d.name != "__pycache__":
                    subdirs.append(f"./datasets/{d.name}")
        self.dataset_dropdown.configure(values=list(set(subdirs)))
        self.dataset_dropdown.set("./datasets/guwahati_azara_processed")
        self.update_direction_preview()


# ==========================================
# PAGE 3: FINETUNING RUN PAGE
# ==========================================
class FinetunePage(ctk.CTkFrame):
    def __init__(self, parent, controller: Pix2PixGUI):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller

        # Hyperparameter States stored internally
        self.fine_dataroot = "./datasets/guwahati_azara_processed"
        self.fine_batch = "4"
        self.fine_epochs = "25"
        self.fine_epochs_decay = "25"
        self.fine_epoch = "latest"
        self.fine_start_count = "51"
        self.checkpoint_epochs = ["latest"]
        self.datasets_list = ["./datasets/guwahati_azara_processed"]

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        back_btn = ctk.CTkButton(
            header, 
            text="⏴ Dashboard", 
            font=FONT_BUTTON, 
            width=130, 
            height=34, 
            corner_radius=17, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1,
            border_color="#cbd5e1",
            command=lambda: self.controller.show_page("home")
        )
        back_btn.pack(side="left")
        
        title = ctk.CTkLabel(header, text="  Finetune Saved Checkpoints", font=FONT_HEADING, text_color="#0f172a")
        title.pack(side="left", padx=10)

        # Settings icon button inside header
        self.settings_btn = ctk.CTkButton(
            header, 
            text="⚙", 
            font=(FONT_FAMILY, 20, "bold"), 
            width=36, 
            height=36, 
            corner_radius=18, 
            fg_color="#f8fafc", 
            text_color="#0f9d58", 
            hover_color="#edf2f7", 
            border_width=1.5,
            border_color="#0f9d58",
            command=self.open_settings_modal
        )
        self.settings_btn.pack(side="right")

        # Form Container
        form = ctk.CTkFrame(self, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#e2e8f0")
        form.pack(fill="both", expand=True, ipady=15)
        
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)

        # 1. SELECT TARGET MODEL
        ctk.CTkLabel(form, text="Select Trained Model / Experiment Folder:", font=FONT_LABEL, text_color="#0f9d58").grid(row=0, column=0, padx=30, pady=(20, 2), sticky="w")
        self.model_dropdown = ctk.CTkOptionMenu(form, width=320, height=35, corner_radius=8, command=self.on_model_selected)
        self.model_dropdown.grid(row=1, column=0, padx=30, pady=2, sticky="w")

        # 2. SELECT DATAROOT DIRECTORY
        ctk.CTkLabel(form, text="Select Dataroot Directory:", font=FONT_LABEL, text_color="#334155").grid(row=0, column=1, padx=30, pady=(20, 2), sticky="w")
        dataset_row = ctk.CTkFrame(form, fg_color="transparent")
        dataset_row.grid(row=1, column=1, padx=30, pady=2, sticky="ew")
        
        self.dataset_dropdown = ctk.CTkOptionMenu(dataset_row, width=220, height=35, corner_radius=8, values=["./datasets/guwahati_azara_processed"], command=self.update_direction_preview)
        self.dataset_dropdown.pack(side="left")
        
        browse_btn = ctk.CTkButton(
            dataset_row, 
            text="Browse...", 
            font=FONT_BUTTON, 
            width=90, 
            height=35, 
            corner_radius=8, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1, 
            border_color="#cbd5e1",
            command=self.browse_dataset_dir
        )
        browse_btn.pack(side="left", padx=5)

        # Static configs
        self.generator_net = "unet_256"
        self.direction = "AtoB"
        self.model_type = "pix2pix"

        # Image Cache to prevent garbage collection
        self.cached_left_img = None
        self.cached_right_img = None

        # Translation preview container
        ctk.CTkLabel(form, text="Image Translation Direction Preview:", font=FONT_LABEL, text_color="#334155").grid(row=2, column=0, columnspan=2, padx=30, pady=(25, 2), sticky="w")
        
        self.preview_container = ctk.CTkFrame(form, fg_color="#f8fafc", corner_radius=12, border_width=1, border_color="#e2e8f0", height=185)
        self.preview_container.grid(row=3, column=0, columnspan=2, padx=30, pady=5, sticky="ew")
        self.preview_container.grid_propagate(False)
        
        self.img_lbl_left = ctk.CTkLabel(self.preview_container, text="No Image", font=(FONT_FAMILY, 10, "bold"), fg_color="#e2e8f0", width=145, height=145, corner_radius=8)
        self.img_lbl_left.pack(side="left", padx=(35, 10), pady=18)
        
        self.arrow_btn = ctk.CTkButton(
            self.preview_container, 
            text="->", 
            font=(FONT_FAMILY, 24, "bold"), 
            text_color="#0f9d58", 
            fg_color="transparent",
            hover_color="#e2e8f0",
            width=50,
            height=50,
            corner_radius=25,
            command=self.toggle_direction
        )
        self.arrow_btn.pack(side="left", padx=20, pady=18)
        
        self.img_lbl_right = ctk.CTkLabel(self.preview_container, text="No Image", font=(FONT_FAMILY, 10, "bold"), fg_color="#e2e8f0", width=145, height=145, corner_radius=8)
        self.img_lbl_right.pack(side="left", padx=(10, 35), pady=18)
        
        self.preview_info_lbl = ctk.CTkLabel(self.preview_container, text="", font=(FONT_FAMILY, 12, "bold"), text_color="#334155")
        self.preview_info_lbl.pack(side="left", padx=20, pady=18)

        # Submit button
        self.submit_btn = ctk.CTkButton(
            self, 
            text="🚀 Launch Fine-tuning Process", 
            font=(FONT_FAMILY, 13, "bold"), 
            height=46, 
            corner_radius=23, 
            fg_color="#0f9d58", 
            hover_color="#0b7a44", 
            text_color="#ffffff",
            command=self.start_finetuning
        )
        self.submit_btn.pack(fill="x", pady=(20, 0))

    def open_settings_modal(self):
        try:
            self.settings_btn.configure(state="disabled")
            FineSettingsModal(self)
        except Exception as e:
            self.settings_btn.configure(state="normal")
            messagebox.showerror("Error", f"Failed to open settings dialog: {e}")
            traceback.print_exc()

    def on_model_selected(self, model_folder: str):
        self.scan_checkpoint_epochs(model_folder)

    def scan_checkpoint_epochs(self, folder: str):
        path = Path("./checkpoints") / folder
        epochs = []
        if path.exists():
            for f in path.iterdir():
                if f.is_file() and f.suffix == ".pth" and "_net_" in f.name:
                    prefix = f.name.split("_net_")[0]
                    if prefix not in epochs:
                        epochs.append(prefix)
                        
        if not epochs:
            epochs = ["latest"]
            
        num_epochs = sorted([int(e) for e in epochs if e.isdigit()])
        str_epochs = sorted([e for e in epochs if not e.isdigit()])
        sorted_epochs = str_epochs + [str(e) for e in num_epochs]
        
        self.checkpoint_epochs = sorted_epochs
        self.fine_epoch = sorted_epochs[0]
        # Auto trigger default guess count
        if self.fine_epoch.isdigit():
            self.fine_start_count = str(int(self.fine_epoch) + 1)
        else:
            if num_epochs:
                self.fine_start_count = str(max(num_epochs) + 1)
            else:
                self.fine_start_count = "51"

    def load_dataset_sample(self):
        try:
            dataroot = self.fine_dataroot
            train_dir = Path(dataroot) / "train"
            if not train_dir.exists():
                return None
            
            # Find first PNG
            pngs = list(train_dir.glob("*.png"))
            if not pngs:
                return None
                
            img = Image.open(pngs[0])
            w, h = img.size
            w2 = int(w / 2)
            
            # Crop A (left) and B (right)
            img_A = img.crop((0, 0, w2, h))
            img_B = img.crop((w2, 0, w, h))
            
            return img_A, img_B
        except Exception as e:
            print(f"Error loading dataset sample: {e}")
            return None

    def toggle_direction(self):
        if self.direction == "AtoB":
            self.direction = "BtoA"
        else:
            self.direction = "AtoB"
        self.update_direction_preview()

    def browse_dataset_dir(self):
        directory = filedialog.askdirectory(initialdir="./datasets")
        if directory:
            try:
                rel = Path(directory).relative_to(Path.cwd())
                path_str = f"./{rel}"
            except ValueError:
                path_str = directory
            
            current_vals = self.dataset_dropdown.cget("values")
            if path_str not in current_vals:
                new_vals = list(current_vals) + [path_str]
                self.dataset_dropdown.configure(values=new_vals)
            self.dataset_dropdown.set(path_str)
            self.fine_dataroot = path_str
            self.update_direction_preview()

    def update_direction_preview(self, choice=None):
        if hasattr(self, "dataset_dropdown"):
            self.fine_dataroot = self.dataset_dropdown.get()
        direction = self.direction
        sample = self.load_dataset_sample()
        
        if sample:
            img_A, img_B = sample
            if direction == "AtoB":
                left_img = img_A
                right_img = img_B
                info_text = "Input: Thermal IR  ->  Output: Visible RGB"
                self.arrow_btn.configure(text="->")
            else:
                left_img = img_B
                right_img = img_A
                info_text = "Input: Visible RGB  <-  Output: Thermal IR"
                self.arrow_btn.configure(text="<-")
                
            self.cached_left_img = ctk.CTkImage(light_image=left_img, size=(135, 135))
            self.cached_right_img = ctk.CTkImage(light_image=right_img, size=(135, 135))
            
            # Configure and keep references attached to widgets to prevent GC
            self.img_lbl_left.configure(image=self.cached_left_img, text="")
            self.img_lbl_left.image = self.cached_left_img
            self.img_lbl_right.configure(image=self.cached_right_img, text="")
            self.img_lbl_right.image = self.cached_right_img
            self.preview_info_lbl.configure(text=info_text)
        else:
            self.cached_left_img = None
            self.cached_right_img = None
            self.img_lbl_left.image = None
            self.img_lbl_right.image = None
            
            if direction == "AtoB":
                self.img_lbl_left.configure(image=None, text="THERMAL IR", fg_color="#e2eafd")
                self.img_lbl_right.configure(image=None, text="VISIBLE RGB", fg_color="#e8f8f0")
                self.preview_info_lbl.configure(text="Input: Thermal IR  ->  Output: Visible RGB")
                self.arrow_btn.configure(text="->")
            else:
                self.img_lbl_left.configure(image=None, text="VISIBLE RGB", fg_color="#e8f8f0")
                self.img_lbl_right.configure(image=None, text="THERMAL IR", fg_color="#e2eafd")
                self.preview_info_lbl.configure(text="Input: Visible RGB  <-  Output: Thermal IR")
                self.arrow_btn.configure(text="<-")

    def start_finetuning(self):
        # Validate checkpoints folder first
        checkpoints_path = Path("./checkpoints") / self.model_dropdown.get()
        if not checkpoints_path.exists():
            messagebox.showerror("Error", f"Checkpoint folder for '{self.model_dropdown.get()}' not found under ./checkpoints/. Train a model first.")
            return

        start_count = int(self.fine_start_count)
        payload = {
            "name": self.model_dropdown.get(),
            "dataroot": self.fine_dataroot,
            "model": self.model_type,
            "direction": self.direction,
            "netG": self.generator_net,
            "dataset_mode": "aligned",
            "norm": "batch",
            "batch_size": int(self.fine_batch),
            "n_epochs": (start_count - 1) + int(self.fine_epochs),
            "n_epochs_decay": int(self.fine_epochs_decay),
            "parent_epoch": self.fine_epoch,
            "epoch_count": start_count,
            "is_finetuning": 1,
            "status": "pending"
        }

        try:
            run_id = db_create_run(payload)
            self.controller.start_training_direct(run_id)
            self.controller.show_page("runs")
            self.controller.runs_page.select_run_directly(run_id)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_show(self):
        folders = []
        path = Path("./checkpoints")
        if path.exists():
            for d in path.iterdir():
                if d.is_dir() and d.name != "web":
                    folders.append(d.name)
        if not folders:
            folders = ["guwahati_azara_pix2pix"]
            
        self.model_dropdown.configure(values=folders)
        self.model_dropdown.set(folders[0])
        self.on_model_selected(folders[0])
        
        # Scan datasets list
        datasets_path = Path("./datasets")
        subdirs = ["./datasets/guwahati_azara_processed"]
        if datasets_path.exists():
            for d in datasets_path.iterdir():
                if d.is_dir() and d.name != "bibtex":
                    subdirs.append(f"./datasets/{d.name}")
        self.datasets_list = list(set(subdirs))
        self.dataset_dropdown.configure(values=self.datasets_list)
        self.dataset_dropdown.set("./datasets/guwahati_azara_processed")
        self.fine_dataroot = "./datasets/guwahati_azara_processed"
        self.update_direction_preview()


# ==========================================
# PAGE 4: RUNS & REAL-TIME LOGS
# ==========================================
class RunsPage(ctk.CTkFrame):
    def __init__(self, parent, controller: Pix2PixGUI):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        back_btn = ctk.CTkButton(
            header, 
            text="⏴ Dashboard", 
            font=FONT_BUTTON, 
            width=130, 
            height=34, 
            corner_radius=17, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1,
            border_color="#cbd5e1",
            command=lambda: self.controller.show_page("home")
        )
        back_btn.pack(side="left")
        
        title = ctk.CTkLabel(header, text="  Live Monitor & Terminal Logs", font=FONT_HEADING, text_color="#0f172a")
        title.pack(side="left", padx=10)

        grid_panel = ctk.CTkFrame(self, fg_color="transparent")
        grid_panel.pack(fill="both", expand=True)
        grid_panel.grid_columnconfigure(0, weight=1)
        grid_panel.grid_columnconfigure(1, weight=2)
        grid_panel.grid_columnconfigure(2, weight=2)
        grid_panel.grid_rowconfigure(0, weight=1)

        # Left: list of runs
        runs_panel = ctk.CTkFrame(grid_panel, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#e2e8f0")
        runs_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(runs_panel, text="Training History", font=(FONT_FAMILY, 14, "bold"), text_color="#0f172a").pack(pady=10)
        
        self.runs_list_scroll = ctk.CTkScrollableFrame(runs_panel)
        self.runs_list_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # Right: Live progression monitor with loading bar and errors
        terminal_panel = ctk.CTkFrame(grid_panel, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#e2e8f0")
        terminal_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.terminal_title = ctk.CTkLabel(terminal_panel, text="Progress Monitor: No run selected", font=(FONT_FAMILY, 14, "bold"), text_color="#0f172a")
        self.terminal_title.pack(pady=10)
        
        # State variables for progression
        self.total_epochs = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.current_losses = {}
        self.error_lines = []

        # Location Display Panel
        self.location_frame = ctk.CTkFrame(terminal_panel, fg_color="#f8fafc", corner_radius=12, border_width=1, border_color="#e2e8f0")
        self.location_frame.pack(fill="x", padx=20, pady=10)
        
        self.loc_log_lbl = ctk.CTkLabel(self.location_frame, text="Log File: None", font=(FONT_FAMILY, 11), text_color="#475569", anchor="w")
        self.loc_log_lbl.pack(fill="x", padx=15, pady=(8, 2))
        
        self.loc_chk_lbl = ctk.CTkLabel(self.location_frame, text="Checkpoint Folder: None", font=(FONT_FAMILY, 11), text_color="#475569", anchor="w")
        self.loc_chk_lbl.pack(fill="x", padx=15, pady=(2, 8))

        # Progress / Loading Bar
        self.progress_frame = ctk.CTkFrame(terminal_panel, fg_color="transparent")
        self.progress_frame.pack(fill="x", padx=20, pady=15)
        
        self.progress_lbl = ctk.CTkLabel(self.progress_frame, text="Progress: 0%", font=(FONT_FAMILY, 13, "bold"), text_color="#0f172a")
        self.progress_lbl.pack(anchor="w", pady=(0, 5))
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=20, corner_radius=10, progress_color="#1a73e8")
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0.0)

        # Progression Stats Area
        self.stats_frame = ctk.CTkFrame(terminal_panel, fg_color="#f8fafc", corner_radius=12, border_width=1, border_color="#e2e8f0")
        self.stats_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.epoch_lbl = ctk.CTkLabel(self.stats_frame, text="Epoch: - / -", font=(FONT_FAMILY, 14, "bold"), text_color="#1e293b", anchor="w")
        self.epoch_lbl.pack(fill="x", padx=20, pady=(15, 5))
        
        self.iter_lbl = ctk.CTkLabel(self.stats_frame, text="Iteration: -", font=(FONT_FAMILY, 12, "bold"), text_color="#475569", anchor="w")
        self.iter_lbl.pack(fill="x", padx=20, pady=2)
        
        self.losses_lbl = ctk.CTkLabel(self.stats_frame, text="Losses:\nWaiting for active connection...", font=(FONT_FAMILY, 12), text_color="#475569", anchor="w", justify="left")
        self.losses_lbl.pack(fill="both", expand=True, padx=20, pady=(5, 15))
        
        self.log_controls = ctk.CTkFrame(terminal_panel, height=50, fg_color="transparent")
        self.log_controls.pack(fill="x", padx=15, pady=10)
        
        self.stop_btn = ctk.CTkButton(
            self.log_controls, 
            text="🛑 Terminate Run", 
            font=FONT_BUTTON, 
            fg_color="#ef4444", 
            hover_color="#dc2626", 
            corner_radius=8,
            state="disabled", 
            text_color="#ffffff",
            command=self.stop_active_run
        )
        self.stop_btn.pack(side="right", padx=10)
        
        self.ws_lbl = ctk.CTkLabel(self.log_controls, text="Status: Idle", text_color="grey")
        self.ws_lbl.pack(side="left", padx=10)
        
        # Col 2: Epoch Visual Preview (Vertical Stack)
        image_preview_panel = ctk.CTkFrame(grid_panel, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#e2e8f0")
        image_preview_panel.grid(row=0, column=2, padx=10, sticky="nsew")
        
        ctk.CTkLabel(image_preview_panel, text="Epoch Visual Preview", font=(FONT_FAMILY, 14, "bold"), text_color="#0f172a").pack(pady=(15, 2))
        
        self.preview_epoch_lbl = ctk.CTkLabel(image_preview_panel, text="No active epoch visuals loaded", font=(FONT_FAMILY, 11), text_color="#64748b")
        self.preview_epoch_lbl.pack(pady=(0, 10))
        
        # 1. Input Image Section
        ctk.CTkLabel(image_preview_panel, text="INPUT IMAGE (real_A)", font=(FONT_FAMILY, 10, "bold"), text_color="#334155").pack(pady=(5, 2))
        self.img_input_frame = ctk.CTkFrame(image_preview_panel, width=220, height=220, corner_radius=8, fg_color="#f8fafc", border_width=1, border_color="#cbd5e1")
        self.img_input_frame.pack(pady=(0, 10))
        self.img_input_frame.pack_propagate(False)
        self.lbl_input_img = ctk.CTkLabel(self.img_input_frame, text="Waiting...", font=(FONT_FAMILY, 10), text_color="#64748b")
        self.lbl_input_img.pack(fill="both", expand=True)
        
        # 2. Expected Image Section
        ctk.CTkLabel(image_preview_panel, text="EXPECTED IMAGE (real_B)", font=(FONT_FAMILY, 10, "bold"), text_color="#334155").pack(pady=(5, 2))
        self.img_target_frame = ctk.CTkFrame(image_preview_panel, width=220, height=220, corner_radius=8, fg_color="#f8fafc", border_width=1, border_color="#cbd5e1")
        self.img_target_frame.pack(pady=(0, 10))
        self.img_target_frame.pack_propagate(False)
        self.lbl_target_img = ctk.CTkLabel(self.img_target_frame, text="Waiting...", font=(FONT_FAMILY, 10), text_color="#64748b")
        self.lbl_target_img.pack(fill="both", expand=True)
        
        # 3. Model Prediction Section
        ctk.CTkLabel(image_preview_panel, text="MODEL PREDICTION (fake_B)", font=(FONT_FAMILY, 10, "bold"), text_color="#334155").pack(pady=(5, 2))
        self.img_pred_frame = ctk.CTkFrame(image_preview_panel, width=220, height=220, corner_radius=8, fg_color="#f8fafc", border_width=1, border_color="#cbd5e1")
        self.img_pred_frame.pack(pady=(0, 15))
        self.img_pred_frame.pack_propagate(False)
        self.lbl_pred_img = ctk.CTkLabel(self.img_pred_frame, text="Waiting...", font=(FONT_FAMILY, 10), text_color="#64748b")
        self.lbl_pred_img.pack(fill="both", expand=True)

    def select_run_directly(self, run_id: int):
        self.controller.active_run_id = run_id
        self.terminal_title.configure(text=f"Progress Monitor: Run #{run_id}")
        
        # Reset state variables
        self.total_epochs = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.current_losses = {}
        self.error_lines = []
        
        # Fetch run details from local DB
        run = db_get_run(run_id)
        if not run:
            return
            
        self.total_epochs = (run.get("n_epochs") or 25) + (run.get("n_epochs_decay") or 25)
        log_file = run.get("log_file", "None")
        name = run.get("name", "None")
        
        self.loc_log_lbl.configure(text=f"Log File: {log_file}")
        self.loc_chk_lbl.configure(text=f"Checkpoint Folder: ./checkpoints/{name}")
        self.progress_bar.set(0.0)
        self.progress_lbl.configure(text="Progress: 0%")
        self.epoch_lbl.configure(text="Epoch: - / -")
        self.iter_lbl.configure(text="Iteration: -")
        self.losses_lbl.configure(text="Loading logs...")
        
        # Set stop button state depending on status
        if run.get("status") == "running":
            self.stop_btn.configure(state="normal")
            self.ws_lbl.configure(text="Status: Running", text_color="green")
        else:
            self.stop_btn.configure(state="disabled")
            self.ws_lbl.configure(text=f"Status: {run.get('status').capitalize()}", text_color="grey")
            
        # Read the log file contents if they exist
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    content = f.read()
                # Parse all lines from log to update progress details
                self.parse_entire_log_text(content)
            except Exception as e:
                self.losses_lbl.configure(text=f"Error reading log file: {e}")
        else:
            self.losses_lbl.configure(text="No log file found.")

    def parse_entire_log_text(self, text: str):
        lines = text.split("\n")
        for line in lines:
            self.parse_single_log_line(line)
        self.update_progress_ui()

    def parse_single_log_line(self, line_str: str):
        line_str = line_str.strip()
        if not line_str:
            return
            
        # Check for error or traceback lines
        lower_line = line_str.lower()
        if "error" in lower_line or "exception" in lower_line or "traceback" in lower_line or "failed" in lower_line:
            self.error_lines.append(line_str)
            if len(self.error_lines) > 5:
                self.error_lines.pop(0)
            
        # Parse progress stats
        epoch_match = re.search(r"\(epoch:\s*(\d+),\s*iters:\s*(\d+)", line_str)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            self.current_iter = int(epoch_match.group(2))
            
            # Extract losses
            loss_part = line_str.split(")")[-1]
            loss_matches = re.findall(r"(\w+):\s*([\d\.]+)", loss_part)
            if loss_matches:
                self.current_losses = {k: float(v) for k, v in loss_matches}
            
        # Also check for "End of epoch" or model saving
        end_epoch_match = re.search(r"End of epoch (\d+)", line_str)
        if end_epoch_match:
            self.current_epoch = int(end_epoch_match.group(1))

    def handle_direct_log_line(self, line_str):
        self.parse_single_log_line(line_str)
        self.update_progress_ui()

    def update_progress_ui(self):
        # Calculate percentage
        if self.total_epochs and self.total_epochs > 0:
            pct = min(1.0, max(0.0, self.current_epoch / self.total_epochs))
        else:
            pct = 0.0
            
        self.progress_bar.set(pct)
        self.progress_lbl.configure(text=f"Progress: {int(pct * 100)}%")
        
        tot_epochs_str = str(self.total_epochs) if self.total_epochs else "?"
        self.epoch_lbl.configure(text=f"Epoch: {self.current_epoch} / {tot_epochs_str}")
        self.iter_lbl.configure(text=f"Iteration: {self.current_iter if self.current_iter else '-'}")
        
        stats_text = ""
        if self.current_losses:
            stats_text += "Current Losses:\n"
            for k, v in self.current_losses.items():
                stats_text += f"  • {k}: {v:.4f}\n"
        else:
            stats_text += "Losses: Waiting for iteration data...\n"
            
        if self.error_lines:
            stats_text += "\nRecent Output Warnings / Errors:\n"
            for err in self.error_lines:
                stats_text += f"  ⚠️ {err}\n"
                
        self.losses_lbl.configure(text=stats_text)
        
        # Load visuals for current epoch
        run = db_get_run(self.controller.active_run_id)
        if run:
            name = run.get("name")
            if name:
                self.load_epoch_visuals(name, self.current_epoch)

    def stop_active_run(self):
        run_id = self.controller.active_run_id
        if not run_id:
            return
        
        try:
            success = self.controller.stop_training_direct(run_id)
            if success:
                messagebox.showinfo("Success", f"Training Run #{run_id} terminated.")
                self.stop_btn.configure(state="disabled")
                self.ws_lbl.configure(text="Status: Stopped", text_color="grey")
            else:
                messagebox.showerror("Error", "Could not stop the active run.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_runs_list(self, runs):
        for widget in self.runs_list_scroll.winfo_children():
            widget.destroy()
            
        for run in runs:
            run_id = run["id"]
            name = run["name"]
            status = run["status"]
            is_fine = run["is_finetuning"]
            
            run_lbl = f"#{run_id} {name}"
            if is_fine:
                run_lbl += " (Finetune)"
            
            color_map = {
                "running": "#10b981",
                "completed": "#2563eb",
                "failed": "#ef4444",
                "stopped": "#f59e0b",
                "pending": "#64748b"
            }
            status_color = color_map.get(status, "grey")

            frame = ctk.CTkFrame(self.runs_list_scroll, fg_color="transparent")
            frame.pack(fill="x", padx=5, pady=4)
            
            sep = ctk.CTkFrame(self.runs_list_scroll, height=1, fg_color="#f1f5f9")
            sep.pack(fill="x", padx=10)

            lbl = ctk.CTkLabel(frame, text=run_lbl, anchor="w", font=(FONT_FAMILY, 11))
            lbl.pack(side="left", padx=5, fill="x", expand=True)
            
            status_tag = ctk.CTkLabel(frame, text=f"● {status.upper()}", font=FONT_LABEL, text_color=status_color)
            status_tag.pack(side="right", padx=10)

            btn = ctk.CTkButton(
                frame, 
                text="View logs", 
                font=FONT_BUTTON, 
                width=75, 
                height=22, 
                corner_radius=4,
                fg_color="#1a73e8",
                text_color="#ffffff",
                hover_color="#155cb4",
                command=lambda rid=run_id: self.select_run_directly(rid)
            )
            btn.pack(side="right", padx=5)

    def load_epoch_visuals(self, run_name: str, epoch: int):
        images_dir = Path("./checkpoints") / run_name / "web" / "images"
        
        # Check files for this epoch
        real_A_path = images_dir / f"epoch{epoch:03d}_real_A.png"
        fake_B_path = images_dir / f"epoch{epoch:03d}_fake_B.png"
        real_B_path = images_dir / f"epoch{epoch:03d}_real_B.png"
        
        # If files don't exist for this epoch, try epoch-1
        if not real_A_path.exists() and epoch > 1:
            epoch = epoch - 1
            real_A_path = images_dir / f"epoch{epoch:03d}_real_A.png"
            fake_B_path = images_dir / f"epoch{epoch:03d}_fake_B.png"
            real_B_path = images_dir / f"epoch{epoch:03d}_real_B.png"
            
        # Try latest if still not found
        if not real_A_path.exists():
            # Find any image files in the directory
            all_real_A = sorted(list(images_dir.glob("*_real_A.png")))
            if all_real_A:
                real_A_path = all_real_A[-1]
                # Extract epoch from filename e.g. epoch001_real_A.png
                epoch_str = real_A_path.name.split("_")[0].replace("epoch", "")
                try:
                    epoch = int(epoch_str)
                except:
                    pass
                fake_B_path = images_dir / f"epoch{epoch:03d}_fake_B.png"
                real_B_path = images_dir / f"epoch{epoch:03d}_real_B.png"
                
        if real_A_path.exists() and fake_B_path.exists() and real_B_path.exists():
            try:
                self.preview_epoch_lbl.configure(text=f"Displaying visuals from Epoch {epoch}", text_color="#10b981")
                
                # Input image
                img_in = Image.open(real_A_path)
                ctk_in = ctk.CTkImage(light_image=img_in, size=(220, 220))
                self.lbl_input_img.configure(image=ctk_in, text="")
                self.lbl_input_img.image = ctk_in  # keep reference
                
                # Ground truth
                img_tgt = Image.open(real_B_path)
                ctk_tgt = ctk.CTkImage(light_image=img_tgt, size=(220, 220))
                self.lbl_target_img.configure(image=ctk_tgt, text="")
                self.lbl_target_img.image = ctk_tgt
                
                # Prediction
                img_pred = Image.open(fake_B_path)
                ctk_pred = ctk.CTkImage(light_image=img_pred, size=(220, 220))
                self.lbl_pred_img.configure(image=ctk_pred, text="")
                self.lbl_pred_img.image = ctk_pred
            except Exception as e:
                print(f"Error loading preview images: {e}")
        else:
            self.preview_epoch_lbl.configure(text="No visuals generated yet for this run", text_color="#64748b")
            self.lbl_input_img.configure(image=None, text="Input (real_A)\nWaiting...")
            self.lbl_target_img.configure(image=None, text="Ground Truth (real_B)\nWaiting...")
            self.lbl_pred_img.configure(image=None, text="Prediction (fake_B)\nWaiting...")

    def on_show(self):
        runs = db_get_all_runs()
        self.update_runs_list(runs)
        if runs and not self.controller.active_run_id:
            self.select_run_directly(runs[0]["id"])


# ==========================================
# PAGE 5: CHECKPOINTS EXPANDED PANEL
# ==========================================
class CheckpointsPage(ctk.CTkFrame):
    def __init__(self, parent, controller: Pix2PixGUI):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        back_btn = ctk.CTkButton(
            header, 
            text="⏴ Dashboard", 
            font=FONT_BUTTON, 
            width=130, 
            height=34, 
            corner_radius=17, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1,
            border_color="#cbd5e1",
            command=lambda: self.controller.show_page("home")
        )
        back_btn.pack(side="left")
        
        title = ctk.CTkLabel(header, text="  Model Storage Navigator", font=FONT_HEADING, text_color="#0f172a")
        title.pack(side="left", padx=10)

        # Scrollable container
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#e2e8f0")
        self.scroll.pack(fill="both", expand=True, padx=5, pady=5)

    def scan_checkpoints(self):
        checkpoints_path = Path("./checkpoints")
        if not checkpoints_path.exists():
            self.populate_checkpoints([])
            return
        
        try:
            results = []
            for subdir in checkpoints_path.iterdir():
                if subdir.is_dir() and subdir.name != "web":
                    files = []
                    for f in subdir.iterdir():
                        if f.is_file() and f.suffix == ".pth":
                            files.append({
                                "filename": f.name,
                                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                                "last_modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                            })
                    results.append({
                        "experiment_name": subdir.name,
                        "directory_path": str(subdir.absolute()),
                        "files": sorted(files, key=lambda x: x["filename"])
                    })
            self.populate_checkpoints(sorted(results, key=lambda x: x["experiment_name"]))
        except Exception as e:
            messagebox.showerror("Error scanning directories", str(e))

    def populate_checkpoints(self, folders):
        for widget in self.scroll.winfo_children():
            widget.destroy()
            
        if not folders:
            ctk.CTkLabel(self.scroll, text="No model folders found under ./checkpoints/", font=FONT_SUBTITLE, text_color="grey").pack(pady=30)
            return

        for folder in folders:
            exp_name = folder["experiment_name"]
            files = folder["files"]
            
            box = ctk.CTkFrame(self.scroll, fg_color="#f8fafc", corner_radius=12, border_width=1, border_color="#e2e8f0")
            box.pack(fill="x", padx=15, pady=10, ipady=5)
            
            ctk.CTkLabel(box, text=f"Experiment Folder: {exp_name}", font=(FONT_FAMILY, 14, "bold"), text_color="#1e3a8a").pack(anchor="w", padx=15, pady=(10, 5))
            
            if not files:
                ctk.CTkLabel(box, text="  No saved model files (.pth) inside.", font=FONT_SUBTITLE, text_color="grey").pack(anchor="w", padx=25, pady=5)
                continue
                
            for f in files:
                fname = f["filename"]
                fsize = f["size_mb"]
                fdate = f["last_modified"]
                
                try:
                    dt = datetime.fromisoformat(fdate)
                    date_lbl = dt.strftime("%b %d, %Y %I:%M %p")
                except:
                    date_lbl = fdate
                    
                row = ctk.CTkFrame(box, fg_color="transparent")
                row.pack(fill="x", padx=25, pady=3)
                
                lbl_txt = f"📦 {fname}  ({fsize} MB)  -  Saved: {date_lbl}"
                ctk.CTkLabel(row, text=lbl_txt, font=(FONT_FAMILY, 11)).pack(side="left", padx=5)
                
                del_btn = ctk.CTkButton(
                    row, 
                    text="Delete", 
                    font=FONT_BUTTON, 
                    fg_color="#ef4444", 
                    hover_color="#dc2626",
                    text_color="#ffffff",
                    width=90, 
                    height=24,
                    corner_radius=4,
                    command=lambda e=exp_name, fn=fname: self.delete_file(e, fn)
                )
                del_btn.pack(side="right", padx=10)

    def delete_file(self, exp_name: str, filename: str):
        if not messagebox.askyesno("Delete Model", f"Are you sure you want to permanently delete {filename}?"):
            return
            
        filepath = Path("./checkpoints") / exp_name / filename
        if not filepath.exists() or not filepath.is_file() or filepath.suffix != ".pth":
            messagebox.showerror("Error", "Checkpoint file not found.")
            return

        try:
            filepath.unlink()
            messagebox.showinfo("Deleted", f"Deleted {filename} successfully.")
            self.scan_checkpoints()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete file: {str(e)}")

    def on_show(self):
        self.scan_checkpoints()


# ==========================================
# PAGE 6: DATA PREPROCESSING PAGE
# ==========================================
class PreprocessPage(ctk.CTkFrame):
    def __init__(self, parent, controller: Pix2PixGUI):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        
        # Get project root dynamically
        self.project_dir = Path(__file__).parent.absolute()
        
        # Process references
        self.process = None
        self.log_thread = None
        
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        back_btn = ctk.CTkButton(
            header, 
            text="⏴ Dashboard", 
            font=FONT_BUTTON, 
            width=130, 
            height=34, 
            corner_radius=17, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1,
            border_color="#cbd5e1",
            command=self.go_back
        )
        back_btn.pack(side="left")
        
        title = ctk.CTkLabel(header, text="  Dataset Preprocessing & Tiling", font=FONT_HEADING, text_color="#0f172a")
        title.pack(side="left", padx=10)
        
        # Form panel
        form = ctk.CTkFrame(self, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#e2e8f0")
        form.pack(fill="both", expand=True, ipady=15)
        
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)
        
        # Row 0: Input Directory
        ctk.CTkLabel(form, text="Raw Satellite Data Folder (Input Directory):", font=FONT_LABEL, text_color="#334155").grid(row=0, column=0, padx=30, pady=(20, 2), sticky="w")
        input_row = ctk.CTkFrame(form, fg_color="transparent")
        input_row.grid(row=1, column=0, padx=30, pady=2, sticky="ew")
        self.inp_input_dir = ctk.CTkEntry(input_row, height=35, corner_radius=8, font=(FONT_FAMILY, 12))
        self.inp_input_dir.pack(side="left", fill="x", expand=True)
        self.inp_input_dir.insert(0, str(self.project_dir / "Guwahati_Azara_dataset"))
        
        browse_input_btn = ctk.CTkButton(
            input_row, 
            text="Browse...", 
            font=FONT_BUTTON, 
            width=90, 
            height=35, 
            corner_radius=8, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1, 
            border_color="#cbd5e1",
            command=self.browse_input_dir
        )
        browse_input_btn.pack(side="left", padx=(10, 0))
        
        # Row 0: Output Directory
        ctk.CTkLabel(form, text="Processed Dataset Path (Output Directory):", font=FONT_LABEL, text_color="#334155").grid(row=0, column=1, padx=30, pady=(20, 2), sticky="w")
        output_row = ctk.CTkFrame(form, fg_color="transparent")
        output_row.grid(row=1, column=1, padx=30, pady=2, sticky="ew")
        self.inp_output_dir = ctk.CTkEntry(output_row, height=35, corner_radius=8, font=(FONT_FAMILY, 12))
        self.inp_output_dir.pack(side="left", fill="x", expand=True)
        self.inp_output_dir.insert(0, str(self.project_dir / "datasets" / "guwahati_azara_processed"))
        
        browse_output_btn = ctk.CTkButton(
            output_row, 
            text="Browse...", 
            font=FONT_BUTTON, 
            width=90, 
            height=35, 
            corner_radius=8, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1, 
            border_color="#cbd5e1",
            command=self.browse_output_dir
        )
        browse_output_btn.pack(side="left", padx=(10, 0))
        
        # Row 1: Parameters (Tile Size, Stride, Max Nodata)
        param_frame = ctk.CTkFrame(form, fg_color="transparent")
        param_frame.grid(row=2, column=0, columnspan=2, padx=30, pady=15, sticky="ew")
        param_frame.grid_columnconfigure(0, weight=1)
        param_frame.grid_columnconfigure(1, weight=1)
        param_frame.grid_columnconfigure(2, weight=1)
        
        # Tile size
        ctk.CTkLabel(param_frame, text="Tile Size (px):", font=FONT_LABEL, text_color="#334155").grid(row=0, column=0, padx=10, pady=2, sticky="w")
        self.inp_tile_size = ctk.CTkEntry(param_frame, width=150, height=35, corner_radius=8, font=(FONT_FAMILY, 12))
        self.inp_tile_size.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        self.inp_tile_size.insert(0, "256")
        
        # Stride
        ctk.CTkLabel(param_frame, text="Stride (overlap helper):", font=FONT_LABEL, text_color="#334155").grid(row=0, column=1, padx=10, pady=2, sticky="w")
        self.inp_stride = ctk.CTkEntry(param_frame, width=150, height=35, corner_radius=8, font=(FONT_FAMILY, 12))
        self.inp_stride.grid(row=1, column=1, padx=10, pady=2, sticky="w")
        self.inp_stride.insert(0, "192")
        
        # Nodata Ratio
        ctk.CTkLabel(param_frame, text="Max No-Data Ratio (0.0 - 1.0):", font=FONT_LABEL, text_color="#334155").grid(row=0, column=2, padx=10, pady=2, sticky="w")
        self.inp_nodata = ctk.CTkEntry(param_frame, width=150, height=35, corner_radius=8, font=(FONT_FAMILY, 12))
        self.inp_nodata.grid(row=1, column=2, padx=10, pady=2, sticky="w")
        self.inp_nodata.insert(0, "0.15")

        # Row 2: Status indicator and Buttons
        action_row = ctk.CTkFrame(form, fg_color="transparent")
        action_row.grid(row=3, column=0, columnspan=2, padx=30, pady=10, sticky="ew")
        
        self.btn_run = ctk.CTkButton(
            action_row, 
            text="🚀 Run Preprocessing", 
            font=FONT_BUTTON, 
            width=180, 
            height=40, 
            corner_radius=20, 
            fg_color="#0284c7",
            hover_color="#0369a1",
            text_color="#ffffff",
            command=self.start_preprocess
        )
        self.btn_run.pack(side="left")
        
        self.btn_stop = ctk.CTkButton(
            action_row, 
            text="⏹ Stop", 
            font=FONT_BUTTON, 
            width=100, 
            height=40, 
            corner_radius=20, 
            fg_color="#ef4444", 
            hover_color="#dc2626", 
            state="disabled",
            text_color="#ffffff",
            command=self.stop_preprocess
        )
        self.btn_stop.pack(side="left", padx=(15, 0))
        
        self.status_lbl = ctk.CTkLabel(action_row, text="Status: Ready", font=(FONT_FAMILY, 12, "bold"), text_color="#475569")
        self.status_lbl.pack(side="right", padx=10)

        # Row 3.5: Progress Loading Bar (on Row 4)
        self.progress_frame = ctk.CTkFrame(form, fg_color="transparent")
        self.progress_frame.grid(row=4, column=0, columnspan=2, padx=30, pady=(10, 5), sticky="ew")
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal", height=10, corner_radius=5, progress_color="#0284c7")
        self.progress_bar.pack(fill="x", side="left", expand=True)
        self.progress_bar.set(0.0)
        
        self.progress_pct_lbl = ctk.CTkLabel(self.progress_frame, text="0%", font=(FONT_FAMILY, 11, "bold"), text_color="#0284c7", width=40)
        self.progress_pct_lbl.pack(side="right", padx=(10, 0))
        
        # Row 4: Log display area
        ctk.CTkLabel(form, text="Standard Output Logs:", font=FONT_LABEL, text_color="#334155").grid(row=5, column=0, columnspan=2, padx=30, pady=(15, 2), sticky="w")
        
        self.log_textbox = ctk.CTkTextbox(form, height=280, corner_radius=12, border_width=1, border_color="#cbd5e1", font=("Courier New", 12))
        self.log_textbox.grid(row=6, column=0, columnspan=2, padx=30, pady=(2, 15), sticky="nsew")
        form.grid_rowconfigure(6, weight=1)

    def browse_input_dir(self):
        dir_path = filedialog.askdirectory(initialdir=self.inp_input_dir.get())
        if dir_path:
            self.inp_input_dir.delete(0, "end")
            self.inp_input_dir.insert(0, dir_path)
            
    def browse_output_dir(self):
        dir_path = filedialog.askdirectory(initialdir=self.inp_output_dir.get())
        if dir_path:
            self.inp_output_dir.delete(0, "end")
            self.inp_output_dir.insert(0, dir_path)
            
    def start_preprocess(self):
        if platform.system() not in ("Linux", "Windows"):
            messagebox.showerror(
                "Platform Not Supported",
                "Dataset Preprocessing is only supported on Linux and Windows due to specialized satellite science and remote sensing library dependencies."
            )
            return

        input_dir = self.inp_input_dir.get().strip()
        output_dir = self.inp_output_dir.get().strip()
        tile_size = self.inp_tile_size.get().strip()
        stride = self.inp_stride.get().strip()
        max_nodata = self.inp_nodata.get().strip()
        
        if not input_dir or not output_dir:
            messagebox.showerror("Error", "Please fill in both input and output directories.")
            return
            
        cmd = [
            sys.executable, "prepare_satellite_dataset.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--tile_size", tile_size,
            "--stride", stride,
            "--max_nodata_ratio", max_nodata
        ]
        
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.insert("end", f"Starting preprocessing process:\nCommand: {' '.join(cmd)}\n\n")
        
        self.progress_bar.set(0.0)
        self.progress_pct_lbl.configure(text="0%")
        self.total_scenes = 1
        self.current_scene = 0
        
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status_lbl.configure(text="Status: Preprocessing...", text_color="#0284c7")
        
        # Prepare platform-safe kwargs
        kwargs = {}
        if platform.system() != "Windows":
            kwargs["preexec_fn"] = os.setsid

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent.absolute()),
                **kwargs
            )
            
            self.log_thread = threading.Thread(target=self.monitor_output, daemon=True)
            self.log_thread.start()
        except Exception as e:
            self.log_textbox.insert("end", f"Failed to start process: {str(e)}\n")
            self.status_lbl.configure(text="Status: Failed to start", text_color="#ef4444")
            self.btn_run.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            
    def monitor_output(self):
        try:
            while True:
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break
                if line:
                    decoded = line.decode("utf-8", errors="ignore")
                    self.after(0, lambda msg=decoded: self.append_log(msg))
                    
            ret_code = self.process.poll()
            if ret_code == 0:
                self.after(0, self.preprocessing_done)
            else:
                self.after(0, lambda: self.preprocessing_failed(f"Process exited with code {ret_code}"))
        except Exception as e:
            self.after(0, lambda: self.preprocessing_failed(str(e)))
            
    def append_log(self, text):
        self.log_textbox.insert("end", text)
        self.log_textbox.see("end")
        
        # Parse progress
        match_total = re.search(r"Found (\d+) unique Landsat scenes", text)
        if match_total:
            self.total_scenes = max(1, int(match_total.group(1)))
            self.current_scene = 0
            self.progress_bar.set(0.0)
            self.progress_pct_lbl.configure(text="0%")
            
        match_scene = re.search(r"Processing scene:", text)
        if match_scene:
            self.current_scene += 1
            progress_val = min(0.95, self.current_scene / self.total_scenes)
            self.progress_bar.set(progress_val)
            self.progress_pct_lbl.configure(text=f"{int(progress_val * 100)}%")
            
        if "Dataset Preprocessing Complete!" in text:
            self.progress_bar.set(1.0)
            self.progress_pct_lbl.configure(text="100%")
        
    def preprocessing_done(self):
        self.status_lbl.configure(text="Status: Completed successfully!", text_color="#10b981")
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        messagebox.showinfo("Success", "Preprocessing completed successfully! Processed tiles are saved.")
        # Trigger an update of datasets options in TrainPage
        self.controller.train_page.on_show()
        
    def preprocessing_failed(self, err_msg):
        self.status_lbl.configure(text="Status: Preprocessing failed", text_color="#ef4444")
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.log_textbox.insert("end", f"\nError: {err_msg}\n")
        messagebox.showerror("Failed", f"Preprocessing failed: {err_msg}")
        
    def stop_preprocess(self):
        if not self.process:
            return
        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)], capture_output=True)
            else:
                import signal
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.status_lbl.configure(text="Status: Stopped by user", text_color="#f59e0b")
            self.log_textbox.insert("end", "\nProcess terminated by user.\n")
            self.btn_run.configure(state="normal")
            self.btn_stop.configure(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Could not stop process: {str(e)}")
            
    def on_show(self):
        if platform.system() not in ("Linux", "Windows"):
            messagebox.showerror(
                "Platform Not Supported",
                "Dataset Preprocessing is only supported on Linux and Windows due to specialized satellite science and remote sensing library dependencies."
            )
            self.after(10, self.go_back)

    def go_back(self):
        self.controller.show_page("home")



# ==========================================
# PAGE 7: TEST MODEL INTERACTIVE INFERENCE
# ==========================================
class TestModelPage(ctk.CTkFrame):
    def __init__(self, parent, controller: Pix2PixGUI):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        
        # ML state
        self.netG = None
        self.current_model_path = None
        self.selected_img_path = None
        self.cached_input_img = None
        self.cached_target_img = None
        
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        back_btn = ctk.CTkButton(
            header, 
            text="⏴ Dashboard", 
            font=FONT_BUTTON, 
            width=130, 
            height=34, 
            corner_radius=17, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1,
            border_color="#cbd5e1",
            command=self.go_back
        )
        back_btn.pack(side="left")
        
        title = ctk.CTkLabel(header, text="  Interactive Model Inference Tester", font=FONT_HEADING, text_color="#0f172a")
        title.pack(side="left", padx=10)
        
        grid_panel = ctk.CTkFrame(self, fg_color="transparent")
        grid_panel.pack(fill="both", expand=True)
        grid_panel.grid_columnconfigure(0, weight=1)
        grid_panel.grid_columnconfigure(1, weight=1)
        grid_panel.grid_rowconfigure(0, weight=1)
        
        # Left: Controls
        controls_panel = ctk.CTkFrame(grid_panel, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#e2e8f0")
        controls_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(controls_panel, text="Control Panel", font=(FONT_FAMILY, 14, "bold"), text_color="#0f172a").pack(pady=(15, 10))
        
        # Model selector
        ctk.CTkLabel(controls_panel, text="Select Model Folder:", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=30, pady=(10, 2))
        self.model_dropdown = ctk.CTkOptionMenu(controls_panel, height=35, corner_radius=8, command=self.on_model_selected)
        self.model_dropdown.pack(fill="x", padx=30, pady=2)
        
        # Epoch Checkpoint Selector
        ctk.CTkLabel(controls_panel, text="Select Checkpoint Suffix:", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=30, pady=(10, 2))
        self.epoch_dropdown = ctk.CTkOptionMenu(controls_panel, height=35, corner_radius=8)
        self.epoch_dropdown.pack(fill="x", padx=30, pady=2)
        
        self.btn_load = ctk.CTkButton(
            controls_panel, 
            text="📥 Load Model Checkpoint", 
            font=FONT_BUTTON, 
            height=38, 
            corner_radius=8, 
            fg_color="#1a73e8", 
            hover_color="#155cb4",
            text_color="#ffffff",
            command=self.load_checkpoint
        )
        self.btn_load.pack(fill="x", padx=30, pady=15)
        
        # Image selector
        ctk.CTkLabel(controls_panel, text="Select Input Image:", font=FONT_LABEL, text_color="#334155").pack(anchor="w", padx=30, pady=(20, 2))
        self.btn_browse = ctk.CTkButton(
            controls_panel, 
            text="📁 Browse Image File...", 
            font=FONT_BUTTON, 
            height=38, 
            corner_radius=8, 
            fg_color="#f1f5f9", 
            text_color="#0f172a", 
            hover_color="#e2e8f0", 
            border_width=1, 
            border_color="#cbd5e1",
            command=self.browse_image
        )
        self.btn_browse.pack(fill="x", padx=30, pady=5)
        
        self.lbl_file_path = ctk.CTkLabel(controls_panel, text="No image selected", font=(FONT_FAMILY, 11), text_color="#64748b", wraplength=400, justify="left")
        self.lbl_file_path.pack(anchor="w", padx=30, pady=(2, 10))
        
        # Checkbox for using Train Mode (for Batchnorm / Dropout look)
        self.var_train_mode = ctk.BooleanVar(value=False)
        self.chk_train_mode = ctk.CTkCheckBox(
            controls_panel, 
            text="Use Train Mode Stats & Dropout", 
            variable=self.var_train_mode, 
            font=(FONT_FAMILY, 11),
            text_color="#475569"
        )
        self.chk_train_mode.pack(anchor="w", padx=30, pady=(0, 15))
        
        # Run Button
        self.btn_predict = ctk.CTkButton(
            controls_panel, 
            text="⚡ Run Prediction / Inference", 
            font=(FONT_FAMILY, 13, "bold"), 
            height=46, 
            corner_radius=23, 
            fg_color="#0f9d58", 
            hover_color="#0b7a44", 
            text_color="#ffffff",
            state="disabled",
            command=self.run_prediction
        )
        self.btn_predict.pack(fill="x", padx=30, pady=25)
        
        self.lbl_status = ctk.CTkLabel(controls_panel, text="Status: Load a model to begin", font=(FONT_FAMILY, 11, "bold"), text_color="#475569")
        self.lbl_status.pack(pady=5)
        
        # Right: Predictions Preview (Vertical Stack)
        preview_panel = ctk.CTkFrame(grid_panel, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#e2e8f0")
        preview_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(preview_panel, text="Inference Previews", font=(FONT_FAMILY, 14, "bold"), text_color="#0f172a").pack(pady=(15, 5))
        
        # 1. Input Image
        ctk.CTkLabel(preview_panel, text="INPUT IMAGE (Thermal IR / Domain A)", font=(FONT_FAMILY, 10, "bold"), text_color="#475569").pack(pady=(5, 2))
        self.frame_in = ctk.CTkFrame(preview_panel, width=220, height=220, corner_radius=8, fg_color="#f8fafc", border_width=1, border_color="#cbd5e1")
        self.frame_in.pack(pady=(0, 10))
        self.frame_in.pack_propagate(False)
        self.lbl_in = ctk.CTkLabel(self.frame_in, text="No Input Image", font=(FONT_FAMILY, 11), text_color="#94a3b8")
        self.lbl_in.pack(fill="both", expand=True)
        
        # 2. Expected Target
        ctk.CTkLabel(preview_panel, text="EXPECTED TARGET (Ground Truth RGB / Domain B)", font=(FONT_FAMILY, 10, "bold"), text_color="#475569").pack(pady=(5, 2))
        self.frame_tgt = ctk.CTkFrame(preview_panel, width=220, height=220, corner_radius=8, fg_color="#f8fafc", border_width=1, border_color="#cbd5e1")
        self.frame_tgt.pack(pady=(0, 10))
        self.frame_tgt.pack_propagate(False)
        self.lbl_tgt = ctk.CTkLabel(self.frame_tgt, text="Ground Truth not available", font=(FONT_FAMILY, 11), text_color="#94a3b8")
        self.lbl_tgt.pack(fill="both", expand=True)
        
        # 3. Model Output
        ctk.CTkLabel(preview_panel, text="MODEL PREDICTION (Generated RGB / Fake B)", font=(FONT_FAMILY, 10, "bold"), text_color="#475569").pack(pady=(5, 2))
        self.frame_out = ctk.CTkFrame(preview_panel, width=220, height=220, corner_radius=8, fg_color="#f8fafc", border_width=1, border_color="#cbd5e1")
        self.frame_out.pack(pady=(0, 15))
        self.frame_out.pack_propagate(False)
        self.lbl_out = ctk.CTkLabel(self.frame_out, text="Prediction pending", font=(FONT_FAMILY, 11), text_color="#94a3b8")
        self.lbl_out.pack(fill="both", expand=True)

    def go_back(self):
        self.controller.show_page("home")
        
    def on_show(self):
        # Scan models directories
        folders = []
        path = Path("./checkpoints")
        if path.exists():
            for d in path.iterdir():
                if d.is_dir() and d.name != "web":
                    folders.append(d.name)
        if not folders:
            folders = ["guwahati_azara_pix2pix"]
            
        self.model_dropdown.configure(values=folders)
        self.model_dropdown.set(folders[0])
        self.on_model_selected(folders[0])
        
    def on_model_selected(self, model_folder: str):
        path = Path("./checkpoints") / model_folder
        epochs = []
        if path.exists():
            for f in path.iterdir():
                if f.is_file() and f.suffix == ".pth" and "net_G" in f.name:
                    prefix = f.name.split("_net_G")[0]
                    if prefix not in epochs:
                        epochs.append(prefix)
        if not epochs:
            epochs = ["latest"]
            
        num_epochs = sorted([int(e) for e in epochs if e.isdigit()])
        str_epochs = sorted([e for e in epochs if not e.isdigit()])
        sorted_epochs = str_epochs + [str(e) for e in num_epochs]
        
        self.epoch_dropdown.configure(values=sorted_epochs)
        self.epoch_dropdown.set(sorted_epochs[0])
        
    def load_checkpoint(self):
        model_folder = self.model_dropdown.get()
        epoch = self.epoch_dropdown.get()
        model_dir = Path("./checkpoints") / model_folder
        checkpoint_path = model_dir / f"{epoch}_net_G.pth"
        
        if not checkpoint_path.exists():
            messagebox.showerror("Error", f"Checkpoint file '{checkpoint_path.name}' not found.")
            return
            
        try:
            # Parse train options to ensure correct network setup
            opt_path = model_dir / "train_opt.txt"
            input_nc = 3
            output_nc = 3
            ngf = 64
            netG_arch = "unet_256"
            norm_type = "batch"
            use_dropout = True
            
            if opt_path.exists():
                try:
                    with open(opt_path, "r") as f:
                        for line in f:
                            if ":" in line:
                                parts = line.strip().split(":")
                                key = parts[0].strip()
                                val_part = parts[1].strip()
                                val_list = val_part.split()
                                val = val_list[0].strip() if val_list else ""
                                
                                if not val:
                                    continue
                                    
                                if key == "input_nc":
                                    input_nc = int(val)
                                elif key == "output_nc":
                                    output_nc = int(val)
                                elif key == "ngf":
                                    ngf = int(val)
                                elif key == "netG":
                                    netG_arch = val
                                elif key == "norm":
                                    norm_type = val
                                elif key == "no_dropout":
                                    use_dropout = not (val.lower() == "true")
                except Exception as parse_err:
                    print(f"Error parsing train_opt.txt: {parse_err}")
            
            import torch
            from models import networks
            
            # Detect device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dev_name = self.device.type.upper()
            if self.device.type == "cuda":
                dev_name += f" ({torch.cuda.get_device_name(0)})"
            
            self.lbl_status.configure(text=f"Status: Loading checkpoint on {dev_name}...", text_color="#1a73e8")
            self.update_idletasks()
            
            self.netG = networks.define_G(
                input_nc=input_nc, 
                output_nc=output_nc, 
                ngf=ngf, 
                netG=netG_arch, 
                norm=norm_type, 
                use_dropout=use_dropout
            )
            
            state_dict = torch.load(str(checkpoint_path), map_location=self.device, weights_only=True)
            if '_metadata' in state_dict:
                del state_dict['_metadata']
            self.netG.to(self.device)
            self.netG.load_state_dict(state_dict)
            self.netG.eval()
            
            self.current_model_path = checkpoint_path
            self.lbl_status.configure(text=f"Status: Model Loaded on {self.device.type.upper()} ({epoch})!", text_color="#10b981")
            
            if self.selected_img_path:
                self.btn_predict.configure(state="normal")
        except Exception as e:
            traceback.print_exc()
            self.lbl_status.configure(text="Status: Failed to load", text_color="#ef4444")
            messagebox.showerror("Error", f"Failed to load checkpoint: {str(e)}")
            
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All Files", "*.*")
            ]
        )
        if not file_path:
            return
            
        self.selected_img_path = file_path
        self.lbl_file_path.configure(text=f"Selected: {Path(file_path).name}")
        
        try:
            # Load input image
            img = Image.open(file_path).convert("RGB")
            w, h = img.size
            
            # Check if this is an aligned pair (512x256)
            if w == h * 2:
                # Left half is Domain A (input), Right half is Domain B (ground truth)
                w2 = h
                self.cached_input_img = img.crop((0, 0, w2, h))
                self.cached_target_img = img.crop((w2, 0, w, h))
                
                # Update expected target preview
                ctk_tgt = ctk.CTkImage(light_image=self.cached_target_img, size=(220, 220))
                self.lbl_tgt.configure(image=ctk_tgt, text="")
                self.lbl_tgt.image = ctk_tgt
            else:
                self.cached_input_img = img
                self.cached_target_img = None
                
                # Reset expected target preview
                self.lbl_tgt.configure(image=None, text="Ground Truth not available\n(Single-channel input)", text_color="#94a3b8")
                
            # Update input preview
            ctk_in = ctk.CTkImage(light_image=self.cached_input_img, size=(220, 220))
            self.lbl_in.configure(image=ctk_in, text="")
            self.lbl_in.image = ctk_in
            
            # Reset prediction preview
            self.lbl_out.configure(image=None, text="Prediction pending", text_color="#94a3b8")
            
            if self.netG:
                self.btn_predict.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image file: {e}")
            
    def run_prediction(self):
        if not self.netG or not self.cached_input_img:
            return
            
        try:
            self.lbl_status.configure(text="Status: Generating prediction...", text_color="#1a73e8")
            self.update_idletasks()
            
            import torch
            import numpy as np
            import torchvision.transforms as transforms
            
            # Setup transform to match input preprocessing in BaseDataset
            transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # Route input tensor to the loaded device (GPU/CPU)
            device = getattr(self, "device", torch.device("cpu"))
            input_tensor = transform(self.cached_input_img).unsqueeze(0).to(device)
            
            # Run prediction
            if hasattr(self, 'var_train_mode') and self.var_train_mode.get():
                self.netG.train()
            else:
                self.netG.eval()

            with torch.no_grad():
                output_tensor = self.netG(input_tensor)
                
            # Denormalize output from [-1, 1] to [0, 1] range and move back to CPU
            output_tensor = (output_tensor.squeeze(0).cpu() + 1.0) / 2.0
            output_tensor = output_tensor.clamp(0, 1)
            
            # Convert tensor to PIL image
            output_numpy = output_tensor.cpu().numpy().transpose(1, 2, 0)
            output_numpy = (output_numpy * 255.0).astype(np.uint8)
            output_img = Image.fromarray(output_numpy)
            
            # Update prediction preview
            ctk_out = ctk.CTkImage(light_image=output_img, size=(220, 220))
            self.lbl_out.configure(image=ctk_out, text="")
            self.lbl_out.image = ctk_out
            
            self.lbl_status.configure(text="Status: Inference completed!", text_color="#10b981")
        except Exception as e:
            traceback.print_exc()
            self.lbl_status.configure(text="Status: Prediction failed", text_color="#ef4444")
            messagebox.showerror("Error", f"Failed to execute prediction: {str(e)}")


if __name__ == "__main__":
    app = Pix2PixGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
