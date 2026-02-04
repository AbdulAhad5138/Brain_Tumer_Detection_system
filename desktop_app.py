import customtkinter as ctk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

# --- Configuration ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TumorDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("NeuroScan Pro - Desktop Edition")
        self.geometry("1100x700")
        
        # Load Model
        self.model = None
        self.status_var = ctk.StringVar(value="Initializing AI Core...")
        threading.Thread(target=self.load_model, daemon=True).start()

        # Layout Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Controls) ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        # Header
        self.logo_label = ctk.CTkLabel(self.sidebar, text="🧠 NeuroScan\nAI Workstation", 
                                     font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # Workflow Controls
        self.btn_upload = ctk.CTkButton(self.sidebar, text="📂 Import MRI Scan", 
                                      command=self.upload_image, 
                                      height=45,
                                      font=("Arial", 14, "bold"))
        self.btn_upload.grid(row=1, column=0, padx=20, pady=20)

        # Parameters Frame
        self.param_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.param_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.lbl_conf = ctk.CTkLabel(self.param_frame, text="Confidence Threshold")
        self.lbl_conf.pack(anchor="w")
        self.slider_conf = ctk.CTkSlider(self.param_frame, from_=0.0, to=1.0, command=self.update_labels)
        self.slider_conf.set(0.25)
        self.slider_conf.pack(fill="x", pady=5)
        self.val_conf = ctk.CTkLabel(self.param_frame, text="25%", font=("Arial", 11))
        self.val_conf.pack(anchor="e")

        self.lbl_opacity = ctk.CTkLabel(self.param_frame, text="Mask Opacity")
        self.lbl_opacity.pack(anchor="w", pady=(15,0))
        self.slider_opacity = ctk.CTkSlider(self.param_frame, from_=0.0, to=1.0, command=self.update_labels)
        self.slider_opacity.set(0.4)
        self.slider_opacity.pack(fill="x", pady=5)
        self.val_opacity = ctk.CTkLabel(self.param_frame, text="40%", font=("Arial", 11))
        self.val_opacity.pack(anchor="e")
        
        # Action Button
        self.btn_run = ctk.CTkButton(self.sidebar, text="⚡ Start Analysis", 
                                   command=self.run_inference, 
                                   fg_color="#e53935", 
                                   hover_color="#b71c1c",
                                   height=50,
                                   font=("Arial", 15, "bold"))
        self.btn_run.grid(row=3, column=0, padx=20, pady=(30, 20))

        # Metrics Panel
        self.metrics_container = ctk.CTkFrame(self.sidebar, fg_color="#1e293b")
        self.metrics_container.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        self.lbl_result_title = ctk.CTkLabel(self.metrics_container, text="REPORT", font=("Arial", 12, "bold"), text_color="gray")
        self.lbl_result_title.pack(pady=(10,5))
        
        self.lbl_tumor_count = ctk.CTkLabel(self.metrics_container, text="Tumors: -", font=("Consolas", 16))
        self.lbl_tumor_count.pack(anchor="w", padx=15)
        
        self.lbl_max_conf = ctk.CTkLabel(self.metrics_container, text="Confidence: -", font=("Consolas", 16))
        self.lbl_max_conf.pack(anchor="w", padx=15, pady=(0,10))


        self.lbl_status = ctk.CTkLabel(self.sidebar, textvariable=self.status_var, font=("Arial", 12), text_color="#94a3b8")
        self.lbl_status.grid(row=11, column=0, padx=20, pady=20)

        # --- Main View Area ---
        self.view_tabs = ctk.CTkTabview(self)
        self.view_tabs.grid(row=0, column=1, sticky="nsew", padx=20, pady=10)
        
        self.tab_dashboard = self.view_tabs.add("Dashboard")
        self.tab_focus = self.view_tabs.add("Analysis")

        # --- Tab 1: Dashboard (Split View) ---
        self.tab_dashboard.columnconfigure(0, weight=1, uniform="g")
        self.tab_dashboard.columnconfigure(1, weight=1, uniform="g")
        self.tab_dashboard.rowconfigure(1, weight=1)

        ctk.CTkLabel(self.tab_dashboard, text="ORIGINAL", font=("Arial", 12, "bold"), text_color="gray").grid(row=0, column=0, pady=5)
        ctk.CTkLabel(self.tab_dashboard, text="AI RESULT", font=("Arial", 12, "bold"), text_color="#3b82f6").grid(row=0, column=1, pady=5)

        # We use a FRAME to hold the label, giving us a stable reference for size
        self.frame_dash_orig = ctk.CTkFrame(self.tab_dashboard, fg_color="#0f172a")
        self.frame_dash_orig.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.lbl_dash_orig = ctk.CTkLabel(self.frame_dash_orig, text="")
        self.lbl_dash_orig.place(relx=0.5, rely=0.5, anchor="center") # Centered, no stretch

        self.frame_dash_res = ctk.CTkFrame(self.tab_dashboard, fg_color="#0f172a")
        self.frame_dash_res.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        self.lbl_dash_res = ctk.CTkLabel(self.frame_dash_res, text="RUN ANALYSIS")
        self.lbl_dash_res.place(relx=0.5, rely=0.5, anchor="center")

        # Bind resize events
        self.frame_dash_orig.bind("<Configure>", lambda e: self.delayed_resize())

        # --- Tab 2: Focus Mode ---
        self.tab_focus.columnconfigure(0, weight=1)
        self.tab_focus.rowconfigure(0, weight=1)
        
        self.frame_focus = ctk.CTkFrame(self.tab_focus, fg_color="#0f172a")
        self.frame_focus.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_focus_res = ctk.CTkLabel(self.frame_focus, text="Detailed Result")
        self.lbl_focus_res.place(relx=0.5, rely=0.5, anchor="center")

        # Internal State
        self.current_image_path = None
        self.current_cv_image = None
        self.current_rgb_image = None
        self.current_result_image = None
        self.resize_timer = None

    def load_model(self):
        try:
            self.model = YOLO('best.pt')
            self.status_var.set("Ready")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)[:20]}...")

    def update_labels(self, _=None):
        self.val_conf.configure(text=f"{int(self.slider_conf.get()*100)}%")
        self.val_opacity.configure(text=f"{int(self.slider_opacity.get()*100)}%")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            self.current_image_path = file_path
            img = cv2.imread(file_path)
            self.current_cv_image = img
            self.current_rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_result_image = None
            
            self.refresh_displays()
            self.status_var.set("Scan Loaded.")

    def run_inference(self):
        if self.model is None or self.current_cv_image is None:
            messagebox.showwarning("System", "Please wait for model or upload an image.")
            return

        self.status_var.set("Segmenting...")
        self.update_idletasks()

        conf = self.slider_conf.get()
        opacity = self.slider_opacity.get()

        results = self.model.predict(self.current_rgb_image, conf=conf, retina_masks=True)
        result = results[0]
        
        annotated_img = self.current_rgb_image.copy()
        count = 0
        max_c = 0.0

        if result.masks:
            count = len(result.masks)
            for i, mask in enumerate(result.masks.xy):
                c = float(result.boxes.conf[i])
                if c > max_c: max_c = c
                pts = np.array(mask, np.int32).reshape((-1, 1, 2))
                overlay = annotated_img.copy()
                cv2.fillPoly(overlay, [pts], (255, 0, 0))
                cv2.addWeighted(overlay, opacity, annotated_img, 1 - opacity, 0, annotated_img)
                cv2.polylines(annotated_img, [pts], True, (255, 0, 0), 2)
                
        elif result.boxes:
             annotated_img = result.plot()
             count = len(result.boxes)
             if count > 0: max_c = float(result.boxes.conf[0])

        self.current_result_image = annotated_img
        self.refresh_displays()
        self.lbl_tumor_count.configure(text=f"Tumors: {count}")
        self.lbl_max_conf.configure(text=f"Confidence: {max_c:.2f}")
        self.status_var.set("Analysis Successful.")
        self.view_tabs.set("Dashboard")

    def delayed_resize(self):
        # Debounce resize to prevent flicker/lag
        if self.resize_timer:
            self.after_cancel(self.resize_timer)
        self.resize_timer = self.after(200, self.refresh_displays)

    def refresh_displays(self):
        if self.current_rgb_image is not None:
            self.fit_image_to_frame(self.current_rgb_image, self.frame_dash_orig, self.lbl_dash_orig)
        
        if self.current_result_image is not None:
            self.fit_image_to_frame(self.current_result_image, self.frame_dash_res, self.lbl_dash_res)
            self.fit_image_to_frame(self.current_result_image, self.frame_focus, self.lbl_focus_res)
        else:
            self.lbl_dash_res.configure(image=None, text="RUN ANALYSIS")

    def fit_image_to_frame(self, img_array, frame, label):
        # robustly get size
        f_w = frame.winfo_width()
        f_h = frame.winfo_height()
        
        if f_w < 50: f_w = 300
        if f_h < 50: f_h = 300
        
        # Add padding by scaling down target dimensions (85% of frame)
        target_w = int(f_w * 0.85)
        target_h = int(f_h * 0.85)

        h, w, _ = img_array.shape
        scale = min(target_w/w, target_h/h)
        
        new_w, new_h = int(w*scale), int(h*scale)
        
        # High-quality resize
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        
        # Force size for CustomTkinter
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
        
        label.configure(image=ctk_img, text="")
        label.image = ctk_img

if __name__ == "__main__":
    app = TumorDetectionApp()
    app.mainloop()
