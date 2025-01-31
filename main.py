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
        self.slider_theta = Slider(plt.axes([0.1, 0.10, 0.35, 0.03]), 'θ (degrees)', 0, 180, valinit=0)
        self.slider_phi = Slider(plt.axes([0.1, 0.15, 0.35, 0.03]), 'ϕ (degrees)', -180, 180, valinit=0)
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

        self.zero_value = 1 + 0j
        self.one_value = 0 + 0j

        self.update(0)

    def get_rotations(self):
        """Returns the theta and phi rotations of the qubit state on the Bloch Sphere."""
        zero_value = self.zero_value  # Amplitude for |0⟩
        one_value = self.one_value    # Amplitude for |1⟩

        # Magnitudes of the complex numbers
        abs_zero = np.abs(zero_value)
        abs_one = np.abs(one_value)

        # Calculate theta (polar angle) based on the magnitude of the |0⟩ state
        theta = 2 * np.arccos(abs_zero)  # theta is 2 * arccos(|zero_value|)

        # Calculate the phase difference between |0⟩ and |1⟩ states
        phi = np.angle(one_value) - np.angle(zero_value)

        return np.rad2deg(theta), np.rad2deg(phi)

    def get_bloch_coordinates(self):
        """Returns the X, Y, Z coordinates of the qubit state on the Bloch Sphere."""
        theta, phi = self.slider_theta.val, self.slider_phi.val

        # Compute the Bloch sphere coordinates
        x = np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        y = np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        z = np.cos(np.deg2rad(theta))

        return x, y, z

    def wavefunction(self):
        """Returns the wavefunction amplitudes for the given theta and phi angles."""
        zero_value = np.cos(np.deg2rad(self.slider_theta.val / 2))
        one_value = np.sin(np.deg2rad(self.slider_theta.val / 2)) * np.exp(1j * np.deg2rad(self.slider_phi.val))

        return zero_value, one_value

    def draw_qubit(self):
        """Draws the qubit state on the Bloch Sphere."""
        x, y, z = self.get_bloch_coordinates()

        # Update the quiver arrow.
        self.quiver_arrow.remove()
        self.quiver_arrow = self.ax_sphere.quiver(0, 0, 0, x, y, z, color='black', arrow_length_ratio=0.1)
        self.qubit_label.remove()
        self.qubit_label = self.ax_sphere.text(x * 1.1, y * 1.1, z * 1.1, '|ψ⟩', color='black', fontsize=12, ha='center', va='center')

        self.ax_sphere.set_title(f"Bloch Sphere\n{np.round(self.zero_value, 2)}∣0⟩\n{np.round(self.one_value, 2)}∣1⟩")

    def update(self, val):
        """Update the plot with the new qubit state."""
        self.draw_qubit()

        # Update wavefunction probabilities
        wf_0, wf_1 = self.wavefunction()
        self.bar[0].set_height(np.abs(wf_0)**2)
        self.bar[1].set_height(np.abs(wf_1)**2)

        # Redraw only the modified elements
        self.fig.canvas.draw_idle()

    def animate_rotations(self):
        theta, phi = self.get_rotations()
        current_theta = self.slider_theta.val
        current_phi = self.slider_phi.val

        if abs(phi - current_phi) > 180:
            phi = phi - np.sign(phi) * 360

        for new_theta, new_phi in zip(np.linspace(current_theta, theta, 20), np.linspace(current_phi, phi, 20)):
            if abs(new_phi) > 180:
                new_phi = new_phi - np.sign(new_phi) * 360

            self.slider_theta.set_val(new_theta)
            self.slider_phi.set_val(new_phi)
            self.update(0)
            plt.pause(0.01)

    def x_gate_animation(self, val):
        """Animate the X-gate operation. Which multiplies the qubit state by the Pauli-X matrix."""
        pauli_x = np.array([[0, 1], [1, 0]])
        components = np.array([self.zero_value, self.one_value])
        new_components = np.dot(pauli_x, components)
        self.zero_value = new_components[0]
        self.one_value = new_components[1]

        self.animate_rotations()

    def y_gate_animation(self, val):
        """Animate the Y-gate operation."""
        pauli_y = np.array([[0, -1j], [1j, 0]])
        components = np.array([self.zero_value, self.one_value])
        new_components = np.dot(pauli_y, components)
        self.zero_value = new_components[0]
        self.one_value = new_components[1]

        self.animate_rotations()

    def z_gate_animation(self, val):
        """Animate the Z-gate operation."""
        pauli_z = np.array([[1, 0], [0, -1]])
        components = np.array([self.zero_value, self.one_value])
        new_components = np.dot(pauli_z, components)
        self.zero_value = new_components[0]
        self.one_value = new_components[1]

        self.animate_rotations()

    def s_gate_animation(self, val):
        """Animate the S-gate operation."""
        phase = np.array([[1, 0], [0, 1j]])
        components = np.array([self.zero_value, self.one_value])
        new_components = np.dot(phase, components)
        self.zero_value = new_components[0]
        self.one_value = new_components[1]

        self.animate_rotations()

    def t_gate_animation(self, val):
        """Animate the T-gate operation."""
        t = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        components = np.array([self.zero_value, self.one_value])
        new_components = np.dot(t, components)
        self.zero_value = new_components[0]
        self.one_value = new_components[1]

        self.animate_rotations()
    
    def h_gate_animation(self, val):
        """Animate the H-gate operation."""
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        components = np.array([self.zero_value, self.one_value])
        new_components = np.dot(h, components)
        self.zero_value = new_components[0]
        self.one_value = new_components[1]

        self.animate_rotations()

    def measure(self, val):
        """Simulate a measurement on the qubit state."""
        wf_0, wf_1 = self.wavefunction()
        prob_0 = np.abs(wf_0) ** 2
        prob_1 = np.abs(wf_1) ** 2

        # Randomly choose a state based on the probabilities
        state = np.random.choice([0, 1], p=[prob_0, prob_1])
        self.y_flip_value = 0

        # Update the wavefunction to the measured state
        if state == 0:
            self.slider_theta.set_val(0)
            self.slider_phi.set_val(0)
            self.one_value = 0 + 0j
            self.zero_value = 1 + 0
        else:
            self.slider_theta.set_val(180)
            self.slider_phi.set_val(0)
            self.one_value = 1 + 0j
            self.zero_value = 0 + 0j


# Initialize plot
sphere = BlochSphere()
plt.draw()
plt.legend()
plt.tight_layout()

plt.show()
