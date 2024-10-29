from utils import delete_files_in_directory
import os
import jax
import jax.numpy as jnp
from jax import jit, device_count
import numpy as np
from time import time
import svgwrite
import io
import matplotlib.pyplot as plt
import soundfile as sf
from xlb.utils import *
from xlb.models import BGKSim
from xlb.lattice import LatticeD2Q9
from xlb.boundary_conditions import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from functools import partial

os.environ["JAX_PLATFORMS"] = "cpu"
jax.config.update('jax_enable_x64', True)

#simulate given a specific geometry and boundary conditions

def make_params(m_per_pixel=1e-4, seconds_of_simulation=0.005, seconds_per_frame_p=1e-4, fluid_flow_velocity=0.01, feature_size_p=0.006):
    """
    Calculate some of the parameters needed for a simulation of a small scale device
    """
    precision = 'f64/f64'
    lattice = LatticeD2Q9(precision)

    #physical unit conversion happens here
    #https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/contrib/D3Q19_lattice_boltzmann_method_unit_conversion.py

    scale = 1.0
    m_per_lattice = m_per_pixel/scale

    # Get conversion factors for length: Δxₚ
    ΔX_P = ΔY_P = ΔZ_P = m_per_lattice

    # The speed of sound in physical and lattice units (_L)
    speed_of_sound_P  = 343              # m/s #change to 343 m/s for real air / doing timing calculations
    speed_of_sound_L  = 1/(jnp.sqrt(3))

    # The density in physical units
    density_P = 1.225 # kg/m³ #density of air, 15 C at sea level

    #keep reynolds number the same, while running the simulation "faster"
    sim_constant = 1
    # Physical parameters in physical units
    KINEMATIC_VISCOSITY_P        = sim_constant * 1.48e-5       # in m2/s
    # max flow at center of parabolic profile #UPDATE THIS TO REFLECT THE TRUE HORIZONTAL INFLOW OR REMOVE
    HORIZONTAL_INFLOW_VELOCITY_P = sim_constant * fluid_flow_velocity          # in m/s

    # Plotting parameters
    SECONDS_OF_SIMULATION = seconds_of_simulation
    ARROW_DENSITY = 15

    # Get conversion factor for time Δtₚ
    # By determining the relevant factor based on Δxₚ and the speed of sound L/P ratio
    ΔT_P_actual = (speed_of_sound_L / 
            speed_of_sound_P * 
            ΔX_P) # in s per lattice time step 
    #comes out to 3.85 * 10^-8 seconds per timestep. That takes a while.

    # or setting Δtₚ, which artificially raises the Mach number
    #ΔT_P              = 1.0e-6 #1.0e-6 #1.6e-4
    ΔT_P = ΔT_P_actual.item()
    print(f"Requested ΔT_P: {ΔT_P_actual}\nSelected ΔT_P: {ΔT_P}\nRatio (selected / requested): {ΔT_P/ΔT_P_actual:.1f}")

    SECONDS_PER_FRAME_P = seconds_per_frame_p #0.02 runs in real time if our gif is 50 fps

    # Get conversion factor mass Δmₚ
    ΔM_P = density_P * (ΔX_P ** 3)

    # Define functions based on conversion factors based on units of variable
    def convert_to_lattice_units(value, length = 0, time = 0, mass = 0):
        return value * (ΔX_P ** -length) * (ΔT_P ** -time) * (ΔM_P ** -mass)

    def convert_to_physical_units(value, length = 0, time = 0, mass = 0):
        return value * (ΔX_P ** length) * (ΔT_P ** time) * (ΔM_P ** mass)

    # Convert physical radius to lattice units
    FEATURE_SIZE_P = feature_size_p# width of the inlet # 0.001 #arbitrary 1 mm feature size
    FEATURE_SIZE_L = convert_to_lattice_units(FEATURE_SIZE_P, length = 1)

    KINEMATIC_VISCOSITY_L        = convert_to_lattice_units(
        KINEMATIC_VISCOSITY_P, 
        length = 2, 
        time = -1)

    HORIZONTAL_INFLOW_VELOCITY_L = convert_to_lattice_units(
        HORIZONTAL_INFLOW_VELOCITY_P,
        length = 1,
        time = -1)

    # Dimensionless constants
    reynolds_number_L = (HORIZONTAL_INFLOW_VELOCITY_L * 2 * FEATURE_SIZE_L) / KINEMATIC_VISCOSITY_L
    reynolds_number_P = (HORIZONTAL_INFLOW_VELOCITY_P * 2 * FEATURE_SIZE_P) / KINEMATIC_VISCOSITY_P

    mach_number_L = HORIZONTAL_INFLOW_VELOCITY_L / speed_of_sound_L
    RELAXATION_OMEGA = float(1.0 / 
                    (KINEMATIC_VISCOSITY_L /
                        (speed_of_sound_L**2) + 
                        0.5)
                    )

    iterations_per_second = 1/ΔT_P
    NUMBER_OF_ITERATIONS = round(iterations_per_second * SECONDS_OF_SIMULATION)

    PLOT_EVERY_N_STEP = int(round(NUMBER_OF_ITERATIONS * SECONDS_PER_FRAME_P / SECONDS_OF_SIMULATION))

    print(f'Number of iterations:      {NUMBER_OF_ITERATIONS}')
    print(f'Lattice Reynolds number:  {reynolds_number_L: g}')
    print(f'Physical Reynolds number: {reynolds_number_P: g}')
    print(f'Horizontal inflow velocity (m/s): {HORIZONTAL_INFLOW_VELOCITY_P: g}')
    print(f'Horizontal inflow velocity lattice: {HORIZONTAL_INFLOW_VELOCITY_L: g}')
    print(f'Mach number:              {mach_number_L: g}')
    print(f'Relaxation time:          {1.0 /RELAXATION_OMEGA: g}')

    print(f'{ΔX_P=} m')
    print(f'{ΔT_P=} s')
    print(f'{ΔM_P=} kg')

    kwargs = {
        'lattice': lattice,
        'omega': RELAXATION_OMEGA,
        #'horizontal_input_velocity':HORIZONTAL_INFLOW_VELOCITY_L,
        'precision': precision,
        'io_rate': PLOT_EVERY_N_STEP,
        'save_vtk':False,
        'print_info_rate': PLOT_EVERY_N_STEP,
        'arrow_density':ARROW_DENSITY,
        'return_fpost': False,  # Set to False if you don't need post-collision fields
        'total_sim_length_steps': NUMBER_OF_ITERATIONS,
        'total_sim_length_p': SECONDS_OF_SIMULATION,
        # add the other physical constants here too
        'ΔX_P': ΔX_P,
        'ΔT_P': ΔT_P,
        'ΔM_P': ΔM_P,

        #'geometry': geometry,
    }
    return kwargs


class SVGSim(BGKSim):
    def __init__(self, **kwargs):
        self.total_sim_length_steps = kwargs['total_sim_length_steps']
        self.total_sim_length_p = kwargs['total_sim_length_p']
        self.save_vtk = kwargs['save_vtk']
        self.ΔX_P = kwargs['ΔX_P']
        self.ΔT_P = kwargs['ΔT_P']
        self.ΔM_P = kwargs['ΔM_P']
        self.arrow_density = kwargs['arrow_density']
        #self.lattice_scale = kwargs['lattice_scale']
        self.geometry = kwargs['geometry']
        self.boundaries = kwargs['boundaries']
        self.color_velocities = kwargs['color_velocities']
        super().__init__(**kwargs)

        #recording (make this a separate class or function or something)
        self.times = []
        self.output_pressures = []
        self.input_pressures = []
        self.input_mass_flows = []
        self.output_mass_flows = []
    
    def convert_to_lattice_units(self, value, length = 0, time = 0, mass = 0):
        return value * (self.ΔX_P ** -length) * (self.ΔT_P ** -time) * (self.ΔM_P ** -mass)

    def convert_to_physical_units(self, value, length = 0, time = 0, mass = 0):
        return value * (self.ΔX_P ** length) * (self.ΔT_P ** time) * (self.ΔM_P ** mass)
    
    def convert_geometry_to_tuple(geometry):
        point_tuples = np.argwhere(geometry).T
        return (tuple(point_tuples[1]), tuple(point_tuples[0]))
        
    def set_boundary_conditions(self):
        """
        Set the boundary conditions for all edges of the domain and all specified geometries.
        """
        #set each of the color velocities
        for color, color_geometry in self.geometry.items():
            #color_geometry = self.geometry[color]
            color_velocity = self.color_velocities[color]
            # find the proper velocity
            x_component_vel = color_velocity['magnitude'] * np.cos(color_velocity['direction'])
            y_component_vel = color_velocity['magnitude'] * np.sin(color_velocity['direction'])

            #fix the tuple problem

            # convert the numpy array to coordinate tuples
            tupled_color_geometry = SVGSim.convert_geometry_to_tuple(color_geometry)
            # set the boundary condition
            vel_geometry = np.zeros_like(tupled_color_geometry, dtype=self.precisionPolicy.compute_dtype).T
            vel_geometry[:, 0] = x_component_vel
            vel_geometry[:, 1] = y_component_vel
            print(vel_geometry)
            self.BCs.append(Regularized(tupled_color_geometry, self.gridInfo, self.precisionPolicy, 'velocity', vel_geometry))

        #Add the boundary conditions for the 
        for boundary_name, boundary_info in self.boundaries.items():
            boundary = self.boundingBoxIndices[boundary_name]
            #default of 'wall' gives no-slip condition
            x_component_vel = 0
            y_component_vel = 0
            if boundary_info['type'] == 'velocity':
                # find the proper velocity
                x_component_vel = boundary_info['magnitude'] * np.cos(boundary_info['direction'])
                y_component_vel = boundary_info['magnitude'] * np.sin(boundary_info['direction'])
            vel_geometry = np.zeros(boundary.shape, dtype=self.precisionPolicy.compute_dtype)
            vel_geometry[:, 0] = x_component_vel
            vel_geometry[:, 1] = y_component_vel
            self.BCs.append(Regularized(tupled_color_geometry, self.gridInfo, self.precisionPolicy, 'velocity', vel_geometry))

    def output_data(self, **kwargs):
        # Extract the fields
        rho = np.array(kwargs['rho'][1:-1, 1:-1])
        u = np.array(kwargs['u'][1:-1, 1:-1, :])
        timestep = kwargs['timestep']

        # Ensure rho has the correct shape by squeezing any singleton dimensions
        if rho.ndim == 3 and rho.shape[-1] == 1:
            rho = rho.squeeze(-1)  # Remove the last dimension if it's 1

        # Calculate the magnitude of the velocity
        u_magnitude = np.linalg.norm(u, axis=2)

        # Print shapes for debugging
        #print(f"rho shape: {rho.shape}")
        #print(f"u_magnitude shape: {u_magnitude.shape}")

        # Align field dimensions for VTK saving
        u_x = u[..., 0]
        u_y = u[..., 1]

        # Check if the shapes match
        if rho.shape != u_x.shape or rho.shape != u_y.shape:
            raise ValueError(f"Field dimension mismatch at timestep {timestep}: "
                            f"rho shape {rho.shape}, u_x shape {u_x.shape}, u_y shape {u_y.shape}")

        # Save the fields for visualization in VTK format
        fields = {"rho": rho, "u_x": u_x, "u_y": u_y}
        if self.save_vtk:
            save_fields_vtk(timestep, fields, prefix=OUTPUT_FOLDER)

        # Save images of the velocity field using the save_image function
        #save_image(timestep, u, prefix=OUTPUT_FOLDER)
        time_in_seconds = timestep / self.total_sim_length_steps * self.total_sim_length_p
        start_image_time = time()
        self.save_image_with_info(timestep, u, prefix=OUTPUT_FOLDER + "u",
                             time_in_seconds=time_in_seconds,
                             vmin=0, vmax=max(u_magnitude.max(), 20*self.horizontal_input_velocity),
                             density=self.arrow_density,
                             cbar_label='Velocity (m/s)'
                             )
        self.save_image_with_info(timestep, rho, prefix=OUTPUT_FOLDER + "rho",
                             time_in_seconds=time_in_seconds,
                             #vmin=0, vmax=2, #max(u_magnitude.max(), 10*self.horizontal_input_velocity),
                             cbar_label='Pressure (PSI)'
                             )
        end_image_time = time()
        print(f"Made the output image in {end_image_time-start_image_time:.2f} seconds.")

        self.times.append(time_in_seconds)

        output_indices = self.boundingBoxIndices['right'].copy()
        output_indices[:, 0] = output_indices[:, 0] - 2
        output_indices = output_indices[2:-2]

        output_indices = tuple(output_indices.T)
        # cur_step_output_pressure = self.scalar_pressure(rho, indices=output_indices)
        # print(f"{cur_step_output_pressure=} PSI")
        # self.output_pressures.append(cur_step_output_pressure)

        input_indices = self.boundingBoxIndices['left'].copy()
        input_indices[:, 0] = input_indices[:, 0] + 2
        input_indices = input_indices[2:-2]
        input_indices = tuple(input_indices.T)
        # cur_step_input_pressure = self.scalar_pressure(rho, indices=input_indices)
        # print(f"{cur_step_input_pressure=} PSI")
        # self.input_pressures.append(cur_step_input_pressure)

        #mass flow (current)
        output_mass_flow = np.sum(self.convert_to_physical_units(rho[output_indices] * u_x[output_indices], time = -1, mass = 1))
        self.output_mass_flows.append(output_mass_flow)
        input_mass_flow = np.sum(self.convert_to_physical_units(rho[input_indices] * u_x[input_indices], time = -1, mass = 1))
        self.input_mass_flows.append(input_mass_flow)

        # TODO:
        # - switch to pressure boundary conditions
        # - Add 'smoke'
        # - record sound pressure at every timestep and play the sound. This could be used as a type of verification, or as a detection of outputs 
        # - time varying inputs
        # - add a length scale on the gif
        # - average pressure over time on the colorbar
        # - median / mean squared / maximum velocity over time plots (in video and in post analysis)
        # - learn to read in vtks for post analysis
        # - learn to do the differentiation stuff

    def scalar_pressure(self, rho, indices=None):
        """
        Convert density in simulation to PSI, assuming the initial
        1 unit of lattice density is 1 atmosphere of pressure (14.7 PSI)
        """
        if not indices:
            #record the default outlet pressure
            indices = self.boundingBoxIndices['right']

        mean_rho = rho[indices].mean()
        atmospheric_pressure_PSI = 14.7
        delta_rho = mean_rho - 1
        delta_p = atmospheric_pressure_PSI * delta_rho
        return delta_p


    def save_image_with_info(self, timestep, fld, prefix=None, time_in_seconds=None,
                             vmin=None, vmax=None, density=1, cbar_label='Magnitude'):
        """
        Save an image of a field at a given timestep with additional information, including vector arrows if the field is 3D.

        Parameters
        ----------
        timestep : int
            The timestep at which the field is being saved.
        fld : jax.numpy.ndarray
            The field to be saved. This should be a 2D or 3D JAX array. If the field is 3D, the magnitude of the field will be calculated and saved.
        prefix : str, optional
            A prefix to be added to the filename. The filename will be the name of the main script file by default.
        time_in_seconds : float, optional
            Time in seconds to be annotated on the image.
        vmin, vmax : float, optional
            The minimum and maximum values for the colorbar. If None, these values will be determined from the data.
        density : int, optional
            The step size for subsampling the vector field when plotting arrows. A lower value means denser arrows.
            A value of 1 means all vectors are plotted; a higher value means fewer vectors are plotted.

        Returns
        -------
        None
        """
        print("Field shape:", fld.shape)
        # Derive the filename from the script name
        # fname = os.path.basename(__main__.__file__)
        # fname = os.path.splitext(fname)[0]
        fname=""
        if prefix is not None:
            fname = prefix + fname
        fname = fname + "_" + str(timestep).zfill(4)

        # Create the plot
        plt.clf()
        dpi = 300
        inch_per_px = 0.05
        fig, ax = plt.subplots()#figsize=(inch_per_px*fld.shape[1] + 2, inch_per_px*fld.shape[0] + 1)) #(fld.shape[1]/dpi + 1, fld.shape[0]/dpi + 1))

        # Check if the field is 3D (vector field) - this is velocity
        if len(fld.shape) == 3 and fld.shape[-1] == 2:
            # Convert velocity components to physical units
            U_physical = self.convert_to_physical_units(fld[..., 0], length=1, time=-1)
            V_physical = self.convert_to_physical_units(fld[..., 1], length=1, time=-1)

            # Calculate the magnitude of the vector field in physical units
            magnitude_physical = np.sqrt(U_physical ** 2 + V_physical ** 2)

            # Convert vmin and vmax to physical units if provided
            if vmin is not None:
                vmin = self.convert_to_physical_units(vmin, length=1, time=-1)
            if vmax is not None:
                vmax = self.convert_to_physical_units(vmax, length=1, time=-1)

            # Plot the magnitude of the vector field
            img = ax.imshow(magnitude_physical.T, cmap=cm.nipy_spectral, origin='lower', vmin=vmin, vmax=vmax)

            # Subsample the vector field based on the density parameter
            step = max(density, 1)
            Y, X = np.meshgrid(np.arange(0, fld.shape[0], step), np.arange(0, fld.shape[1], step))

            U_sampled = U_physical[::step, ::step]
            V_sampled = V_physical[::step, ::step]

            # Normalize the vectors to ensure the maximum length is equal to 'density'
            max_magnitude = np.sqrt(U_sampled**2 + V_sampled**2).max()

            scale_factor = 1.1 * max_magnitude / density if max_magnitude != 0 else 1.0

            # Plot vector arrows with thinner lines and minimum length scaling
            ax.quiver(
                Y, X, U_sampled.T, V_sampled.T, 
                scale=scale_factor, 
                color='white', 
                scale_units='xy', 
                width=0.002,
                minshaft=2,  # Minimum shaft length to make short arrows visible
                capstyle="round"
            )
            #clear all tick marks
            ax.set_xticks([])
            ax.set_yticks([])

        #this is pressure
        elif len(fld.shape) == 2:
            # Convert scalar field to physical units - pressure
            #fld_physical = self.convert_to_physical_units(fld, length=1)
            #fld_physical = self.convert_to_physical_units(fld, length=-1, time=-2, mass=1)

            atmospheric_pressure_PSI = 14.7
            delta_rho = fld - 1
            delta_p = atmospheric_pressure_PSI * delta_rho

            # Convert vmin and vmax to physical units if provided
            # if vmin is not None:
            #     #vmin = self.convert_to_physical_units(vmin, length=1)
            #     vmin = self.convert_to_physical_units(vmin, length=-1, time=-2, mass=1)
            # if vmax is not None:
            #     vmax = self.convert_to_physical_units(vmax, length=-1, time=-2, mass=1)

            # For scalar fields, simply plot the field
            # img = ax.imshow(fld_physical.T, cmap=cm.nipy_spectral, origin='lower', vmin=vmin, vmax=vmax)
            # img = ax.imshow(fld.T, cmap=cm.nipy_spectral, origin='lower', vmin=vmin, vmax=vmax)
            vmin = 0
            vmax = 0.02 #atmospheric_pressure_PSI * 2
            img = ax.imshow(delta_p.T, cmap=cm.nipy_spectral, origin='lower', vmin=vmin, vmax=vmax)
            
            #clear all tick marks
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            raise ValueError("The input field should either be a 2D scalar field or a 3D vector field with two components.")

        # Make an axes divider and append a colorbar to the right of the image, sized to match the image height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.set_label(cbar_label, fontsize=6)

        # Ensure the colorbar ticks do not use scientific notation
        cbar.formatter = FormatStrFormatter('%.4f')
        cbar.update_ticks()

        # Add time annotation, if provided
        if time_in_seconds is not None:
            time_str = f"Time: {time_in_seconds:.4f} s"
            ax.text(0.7, 0.95, time_str, transform=ax.transAxes, fontsize=6, verticalalignment='top', color='white')

        # Save the image
        fig.savefig(fname + '.png', bbox_inches='tight', dpi=dpi) #
        plt.close(fig)


def run_simulation(geometries, json_data, output_folder = 'simulation_outputs/test_sim/', **kwargs):
    """
    Runs the simulation using the geometries and boundary conditions.

    Returns:
    - results: Simulation results (modify as needed).
    """
    # Access boundary conditions and color velocities
    boundaries = json_data['boundaries']
    color_velocities = json_data['colorVelocities']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    delete_files_in_directory(output_folder)

    max_input_flow_velocity = max([flow_info["magnitude"] for color_name, flow_info in color_velocities.items()])
    params = make_params(m_per_pixel=1e-4, seconds_of_simulation=0.005, seconds_per_frame_p=1e-4,
                         fluid_flow_velocity=max_input_flow_velocity, feature_size_p=0.006)
    # params['geometry'] = geometries
    # params['boundaries'] = boundaries
    # params['color_velocities'] = color_velocities
    first_geometry = list(geometries.values())[0]
    NX = first_geometry.shape[1]
    NY = first_geometry.shape[0]
    params.update({
        'geometry':geometries,
        'boundaries':boundaries,
        'color_velocities':color_velocities,
        'nx': NX,
        'ny': NY,
        'nz': 0,
    })

    sim_start = time()
    print("here")
    sim = SVGSim(**params)
    sim.run(params['total_sim_length_steps'])
    sim_end = time()
    print(f"Total elapsed time: {sim_end - sim_start:.2f} seconds")

    # Placeholder for simulation results
    results = {}
    return results
