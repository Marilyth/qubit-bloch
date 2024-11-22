import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import cm

class BlochSphere:
    """A class to visualize a single qubit state on the Bloch Sphere.
    """

    def __init__(self) -> None:
        self.y_flip_value = 0

        # Plot setup
        self.fig = plt.figure(figsize=(10, 5))

        # Bloch Sphere
        self.ax_sphere = self.fig.add_subplot(121, projection='3d')
        self.ax_sphere.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        self.ax_sphere.set_xlim([-1, 1])
        self.ax_sphere.set_ylim([-1, 1])
        self.ax_sphere.set_zlim([-1, 1])
        self.ax_sphere.set_xlabel('X')
        self.ax_sphere.set_ylabel('Y')
        self.ax_sphere.set_zlabel('Z')

        # Setup Probability plot.
        ax_wave = self.fig.add_subplot(122)
        ax_wave.set_ylim([0, 1])
        ax_wave.set_title("Probabilities")
        ax_wave.set_ylabel("Probability")
        self.bar = ax_wave.bar(['|0⟩', '|1⟩'], [0.5, 0.5], color=['#FF000033', '#0000FF33'])

        # Precompute the sphere surface for optimization
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax_sphere.plot_surface(x, y, z, color='cyan', alpha=0.1, rstride=7, cstride=7, cmap=cm.coolwarm, edgecolor='gray')
        self.ax_sphere.set_axis_off()

        # Setup sphere axes.
        self.ax_sphere.quiver(0, 0, 0, 1.3, 0, 0, color='#FF000033', label='X', arrow_length_ratio=0.1)
        self.ax_sphere.text(1.4, 0, 0, 'X', color='#FF000033', fontsize=12, ha='center', va='center')
        self.ax_sphere.quiver(0, 0, 0, 0, 1.3, 0, color='#00FF0033', label='Y', arrow_length_ratio=0.1)
        self.ax_sphere.text(0, 1.4, 0, 'Y', color='#00FF0033', fontsize=12, ha='center', va='center')
        self.ax_sphere.quiver(0, 0, 0, 0, 0, 1.3, color='#0000FF33', label='Z', arrow_length_ratio=0.1)
        self.ax_sphere.text(0, 0, 1.4, 'Z', color='#0000FF33', fontsize=12, ha='center', va='center')

        # Initialize quiver for the qubit arrow.
        self.quiver_arrow = self.ax_sphere.quiver(0, 0, 0, 0, 0, 0, color='black', arrow_length_ratio=0.1)
        self.qubit_label = self.ax_sphere.text(0, 0, 0, '|ψ⟩', color='black', fontsize=12, ha='center', va='center')

        # UI elements, Position: [left, bottom, width, height]
        self.slider_theta = Slider(plt.axes([0.1, 0.10, 0.35, 0.03]), 'θ (degrees)', 0, 360, valinit=0)
        self.slider_phi = Slider(plt.axes([0.1, 0.15, 0.35, 0.03]), 'ϕ (degrees)', 0, 360, valinit=0)
        self.x_gate_button = Button(plt.axes([0.1, 0.20, 0.05, 0.03]), 'X')
        self.y_gate_button = Button(plt.axes([0.15, 0.20, 0.05, 0.03]), 'Y')
        self.z_gate_button = Button(plt.axes([0.2, 0.20, 0.05, 0.03]), 'Z')
        self.h_gate_button = Button(plt.axes([0.25, 0.20, 0.05, 0.03]), 'H')
        self.s_gate_button = Button(plt.axes([0.3, 0.20, 0.05, 0.03]), 'S')
        self.t_gate_button = Button(plt.axes([0.35, 0.20, 0.05, 0.03]), 'T')
        self.measure_button = Button(plt.axes([0.1, 0.25, 0.05, 0.03]), 'M')

        # Connect sliders to update function
        self.slider_theta.on_changed(self.update)
        self.slider_phi.on_changed(self.update)
        self.x_gate_button.on_clicked(self.x_gate_animation)
        self.y_gate_button.on_clicked(self.y_gate_animation)
        self.z_gate_button.on_clicked(self.z_gate_animation)
        self.h_gate_button.on_clicked(self.h_gate_animation)
        self.s_gate_button.on_clicked(self.s_gate_animation)
        self.t_gate_button.on_clicked(self.t_gate_animation)
        self.measure_button.on_clicked(self.measure)

        self.update(0)

    def wavefunction(self, theta, phi):
        """Returns the |0⟩ and |1⟩ amplitudes of a qubit state given theta and phi angles in degrees."""
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        wf_0 = np.cos(theta_rad / 2)
        wf_1 = np.exp(1j * phi_rad) * np.sin(theta_rad / 2)

        if self.y_flip_value != 0:
            # Interpolate toward the Y-gate result
            relative_wf0 = (1 - self.y_flip_value) * wf_0 + (wf_1 * -1j) * self.y_flip_value
            relative_wf1 = (1 - self.y_flip_value) * wf_1 + (wf_0 * 1j) * self.y_flip_value

            # Normalize to ensure unit length
            norm = np.sqrt(np.abs(relative_wf0)**2 + np.abs(relative_wf1)**2)
            wf_0 = relative_wf0 / norm
            wf_1 = relative_wf1 / norm

        return wf_0, wf_1

    def get_bloch_coordinates(self):
        """Returns the X, Y, Z coordinates of the qubit state on the Bloch Sphere."""
        wf_0, wf_1 = self.wavefunction(self.slider_theta.val, self.slider_phi.val)

        # Magnitude of the amplitude
        abs_wf_0 = np.abs(wf_0)

        # Calculate theta
        theta = 2 * np.arccos(abs_wf_0)

        # Calculate the phases of wf_0 and wf_1
        phi_0 = np.angle(wf_0)  # phase of wf_0
        phi_1 = np.angle(wf_1)  # phase of wf_1

        # Calculate the relative phase phi
        phi = phi_1 - phi_0

        # Compute the Bloch sphere coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return x, y, z

    def draw_qubit(self):
        """Draws the qubit state on the Bloch Sphere."""
        x, y, z = self.get_bloch_coordinates()

        # Update the quiver arrow.
        self.quiver_arrow.remove()
        self.quiver_arrow = self.ax_sphere.quiver(0, 0, 0, x, y, z, color='black', arrow_length_ratio=0.1)
        self.qubit_label.remove()
        self.qubit_label = self.ax_sphere.text(x * 1.1, y * 1.1, z * 1.1, '|ψ⟩', color='black', fontsize=12, ha='center', va='center')

        quantum_state = self.wavefunction(self.slider_theta.val, self.slider_phi.val)
        self.ax_sphere.set_title(f"Bloch Sphere\n{np.round(quantum_state[0], 2)}∣0⟩\n{np.round(quantum_state[1], 2)}∣1⟩")

    def update(self, val):
        """Update the plot with the new qubit state."""
        self.draw_qubit()

        # Update wavefunction probabilities
        wf_0, wf_1 = self.wavefunction(self.slider_theta.val, self.slider_phi.val)
        self.bar[0].set_height(np.abs(wf_0)**2)
        self.bar[1].set_height(np.abs(wf_1)**2)

        # Redraw only the modified elements
        self.fig.canvas.draw_idle()

    def x_gate_animation(self, val):
        """Animate the X-gate operation."""
        initial_theta = self.slider_theta.val

        for theta in np.linspace(0, 180, 50):
            self.slider_theta.set_val((initial_theta + theta) % 360)
            self.update(0)
            plt.pause(0.01)

    def y_gate_animation(self, val):
        """Animate the Y-gate operation."""
        initial_y_flip = -self.y_flip_value

        for y_flip in np.linspace(abs(initial_y_flip), initial_y_flip + 1, 50):
            self.y_flip_value = y_flip
            self.update(0)
            plt.pause(0.01)

        self.update(0)

    def z_gate_animation(self, val):
        """Animate the Z-gate operation."""
        initial_phi = self.slider_phi.val

        for phi in np.linspace(0, 180, 50):
            self.slider_phi.set_val((initial_phi + phi) % 360)
            self.update(0)
            plt.pause(0.01)

    def s_gate_animation(self, val):
        """Animate the S-gate operation."""
        initial_phi = self.slider_phi.val

        for phi in np.linspace(0, 90, 25):
            self.slider_phi.set_val((initial_phi + phi) % 360)
            self.update(0)
            plt.pause(0.01)

    def t_gate_animation(self, val):
        """Animate the T-gate operation."""
        initial_phi = self.slider_phi.val

        for phi in np.linspace(0, 45, 12):
            self.slider_phi.set_val((initial_phi + phi) % 360)
            self.update(0)
            plt.pause(0.01)
    
    def h_gate_animation(self, val):
        """Animate the H-gate operation."""
        initial_theta = self.slider_theta.val

        for theta in np.linspace(0, 90, 25):
            self.slider_theta.set_val((initial_theta + theta) % 360)
            self.update(0)
            plt.pause(0.01)

    def measure(self, val):
        """Simulate a measurement on the qubit state."""
        wf_0, wf_1 = self.wavefunction(self.slider_theta.val, self.slider_phi.val)
        prob_0 = np.abs(wf_0) ** 2
        prob_1 = np.abs(wf_1) ** 2

        # Randomly choose a state based on the probabilities
        state = np.random.choice([0, 1], p=[prob_0, prob_1])
        self.y_flip_value = 0

        # Update the wavefunction to the measured state
        if state == 0:
            self.slider_theta.set_val(0)
            self.slider_phi.set_val(0)
        else:
            self.slider_theta.set_val(180)
            self.slider_phi.set_val(0)

# Initialize plot
sphere = BlochSphere()
plt.draw()
plt.legend()
plt.tight_layout()

plt.show()
