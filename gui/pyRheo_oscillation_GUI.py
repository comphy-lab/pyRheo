import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyRheo.oscillation_model import OscillationModel  # Import your updated OscillationModel class
import io
import sys

class CreepApp:
    def __init__(self, root):
        self.root = root
        self.root.title("pyRheo Oscillation Model Fitting")
        self.root.geometry("800x600")

        # Define styles
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 12), padding=5)
        style.configure("TButton", font=("Helvetica", 12), padding=5, relief="flat")
        style.configure("TEntry", font=("Helvetica", 12), padding=5)
        style.configure("TCombobox", font=("Helvetica", 12), padding=5)

        self.root.configure(bg="#f0f4f7")

        # Model selection dropdown using Combobox
        self.model_var = tk.StringVar(root)
        self.model_var.set("FractionalMaxwell")  # Default model
        self.models = [
            "Maxwell", "SpringPot", "FractionalMaxwellGel", "FractionalMaxwellLiquid",
            "FractionalMaxwell", "FractionalKelvinVoigtS", "FractionalKelvinVoigtD",
            "FractionalKelvinVoigt", "Zener", "FractionalZenerSolidS", "FractionalZenerLiquidS",
            "FractionalZenerLiquidD", "FractionalZenerS"
        ]
        ttk.Label(root, text="Select Model:", background="#f0f4f7").pack(pady=5)
        self.model_combobox = ttk.Combobox(root, textvariable=self.model_var, values=self.models, state="readonly")
        self.model_combobox.config(width=25)
        self.model_combobox.pack()

        # Initial guesses input
        ttk.Label(root, text="Number of Initial Guesses:", background="#f0f4f7").pack(pady=5)
        self.initial_guesses_entry = ttk.Entry(root)
        self.initial_guesses_entry.insert(0, "10")  # Default value
        self.initial_guesses_entry.pack()

        # Minimization algorithm dropdown using Combobox
        self.algorithm_var = tk.StringVar(root)
        self.algorithm_var.set("Powell")  # Default algorithm
        algorithms = ["Powell", "Nelder-Mead", "L-BFGS-B", "CG", "BFGS"]
        ttk.Label(root, text="Minimization Algorithm:", background="#f0f4f7").pack(pady=5)
        self.algorithm_combobox = ttk.Combobox(root, textvariable=self.algorithm_var, values=algorithms, state="readonly")
        self.algorithm_combobox.config(width=25)
        self.algorithm_combobox.pack()

        # Create a frame for buttons
        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(pady=10)

        # Load data and Run model buttons
        self.load_button = ttk.Button(self.button_frame, text="Load Data", command=self.load_data)
        self.load_button.grid(row=0, column=0, padx=10, pady=10)

        self.run_button = ttk.Button(self.button_frame, text="Run Model", command=self.run_model)
        self.run_button.grid(row=0, column=1, padx=10, pady=10)

        # Download results button
        self.download_button = ttk.Button(self.button_frame, text="Download Results", command=self.download_results, state="disabled")
        self.download_button.grid(row=1, column=0, padx=10, pady=10)

        # Execute plot button
        self.plot_button = ttk.Button(self.button_frame, text="Plot Results", command=self.plot_results, state="disabled")
        self.plot_button.grid(row=1, column=1, padx=10, pady=10)

        # Execution time display
        self.execution_time_label = ttk.Label(root, text="Execution time: ", background="#f0f4f7", font=("Helvetica", 12, "italic"))
        self.execution_time_label.pack(pady=5)

        # Create a frame for the model parameters and model error
        self.results_frame = ttk.Frame(root)
        self.results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=5)

        # Frame for model parameters display
        self.param_frame = ttk.LabelFrame(self.results_frame, text="Model Parameters", padding=5)
        self.param_frame.pack(fill="both", expand=True)
        self.param_text = tk.Text(self.param_frame, height=6, font=("Helvetica", 12), wrap="word")
        self.param_text.pack(fill="both", expand=True)

        # Frame for error display
        self.error_frame = ttk.LabelFrame(self.results_frame, text="Model Error", padding=5)
        self.error_frame.pack(fill="both", expand=True)
        self.error_text = tk.Text(self.error_frame, height=4, font=("Helvetica", 12), wrap="word")
        self.error_text.pack(fill="both", expand=True)

        # Create a frame for the plot
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=5)

        self.canvas = None

        # Initialize data attributes
        self.omega = None
        self.G_prime = None
        self.G_double_prime = None
        self.model = None

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        self.root.quit()
        self.root.destroy()

    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if file_path:
            try:
                data = pd.read_csv(file_path, delimiter='\t')
                self.omega = data['Angular Frequency'].values
                self.G_prime = data['Storage Modulus'].values
                self.G_double_prime = data['Loss Modulus'].values
                messagebox.showinfo("Data Load", "Data loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")

    def run_model(self):
        if self.omega is None or self.G_prime is None:
            messagebox.showwarning("No Data", "Please load data before running the model.")
            return

        try:
            selected_model = self.model_var.get()
            initial_guesses = int(self.initial_guesses_entry.get())
            algorithm = self.algorithm_var.get()

            start_time = time.time()

            self.model = OscillationModel(
                model=selected_model,
                initial_guesses="random",
                num_initial_guesses=initial_guesses,
                minimization_algorithm=algorithm
            )
            self.model.fit(self.omega, self.G_prime, self.G_double_prime)

            end_time = time.time()
            execution_time = end_time - start_time
            self.execution_time_label.config(text=f"Execution time: {execution_time:.4f} seconds")

            param_output = io.StringIO()
            error_output = io.StringIO()
            sys.stdout = param_output
            self.model.print_parameters()
            sys.stdout = error_output
            self.model.print_error()
            sys.stdout = sys.__stdout__

            self.param_text.delete("1.0", tk.END)
            self.param_text.insert(tk.END, param_output.getvalue())
            self.error_text.delete("1.0", tk.END)
            self.error_text.insert(tk.END, error_output.getvalue())

            self.download_button.config(state="normal")
            self.plot_button.config(state="normal")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")

    def plot_results(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please run the model before plotting results.")
            return

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(self.omega, self.G_prime, label="G_prime", color="dodgerblue")
        ax.scatter(self.omega, self.G_double_prime, label="G_double_prime", color="black")
        predicted_G_prime, predicted_G_double_prime  = self.model.predict(self.omega)
        ax.plot(self.omega, predicted_G_prime, label="G_prime fit", linestyle="-", color="orangered")
        ax.plot(self.omega, predicted_G_double_prime, label="G_double_prime fit", linestyle="--", color="orangered")

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("angular frequency", fontsize=14)
        ax.set_ylabel("storage and loss moduli", fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def download_results(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please run the model before downloading results.")
            return

        predicted_G_prime, predicted_G_double_prime = self.model.predict(self.omega)

        results_df = pd.DataFrame({
            "Angular Frequency": self.omega,
            "Storage Modulus": predicted_G_prime,
            "Loss Modulus": predicted_G_double_prime
        })

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            title="Save Results"
        )

        if save_path:
            try:
                results_df.to_csv(save_path, index=False)
                messagebox.showinfo("Download Complete", f"Results saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CreepApp(root)
    root.mainloop()

