"""
Physical system model.

This module contains two classes:

1. PhysicalWirelessClient
   - Models the time-varying wireless communication rate of one client.
   - Returns the communication rate in bit/s.

2. PhysicalSystem
   - Creates and manages K physical clients.
   - Each client has one wireless object.
   - Calling the system advances all clients by one time step and returns communication rates for all clients
"""

import numpy as np


# =============================================================================
# Bessel function J0
# =============================================================================
try:
    from scipy.special import j0 as bessel_j0
except Exception:
    def bessel_j0(x):
        """Lightweight approximation of the zero-order Bessel function J0(x)."""
        x = np.asarray(x, dtype=float)
        ax = np.abs(x)

        y = (x / 2.0) ** 2
        series = 1.0 - y + y**2 / 4.0 - y**3 / 36.0 + y**4 / 576.0

        asymptotic = np.sqrt(2.0 / (np.pi * np.maximum(ax, 1e-12))) * np.cos(
            ax - np.pi / 4.0
        )

        return np.where(ax < 8.0, series, asymptotic)


# =============================================================================
# Wireless communication client
# =============================================================================
class PhysicalWirelessClient:
    """
    Physical wireless communication model for one mobile client.

    The model includes:
        - mobility inside a circular cell
        - distance-dependent path loss
        - spatially correlated log-normal shadowing
        - time-correlated Rayleigh small-scale fading
        - slowly varying transmit power
        - thermal noise plus receiver noise figure

    Calling the object once advances the wireless state by one time step and
    returns the communication rate in bit/s.

    The returned rate is the rate over one physical bandwidth unit whose size is
    given by ``rb_bandwidth_hz``.
    """

    def __init__(
        self,
        client_id,
        cell_radius=500.0,
        rb_bandwidth_hz=1e6,
        carrier_frequency_hz=3.5e9,
        bs_position=(0.0, 0.0),
        delta_t=1.0,
        rate_efficiency=0.65,
        d0=1.0,
        path_loss_exponent=3.2,
        shadowing_std_db=8.0,
        shadowing_corr_distance_m=30.0,
        noise_density_dbm_per_hz=-174.0,
        mean_speed_range_mps=(0.2, 3.0),
        speed_std_range_mps=(0.05, 0.5),
        speed_correlation_range=(0.80, 0.98),
        direction_change_std_rad=0.2,
        mean_tx_power_range_dbm=(15.0, 23.0),
        tx_power_std_range_db=(0.5, 2.0),
        tx_power_correlation_range=(0.90, 0.99),
        tx_power_min_dbm=10.0,
        tx_power_max_dbm=23.0,
        noise_figure_range_db=(5.0, 10.0),
        noise_variation_std_db=0.5,
        rng=None,
    ):
        self.client_id = client_id
        self.rng = np.random.default_rng(rng)

        self.c = 299_792_458.0

        self.cell_radius = float(cell_radius)
        self.rb_bandwidth_hz = float(rb_bandwidth_hz)
        self.carrier_frequency_hz = float(carrier_frequency_hz)
        self.bs_position = np.array(bs_position, dtype=float)
        self.delta_t = float(delta_t)
        self.rate_efficiency = float(rate_efficiency)
        self.d0 = float(d0)
        self.eta = float(path_loss_exponent)
        self.sigma_X = float(shadowing_std_db)
        self.shadowing_corr_distance_m = float(shadowing_corr_distance_m)
        self.noise_density_dbm_per_hz = float(noise_density_dbm_per_hz)
        self.direction_change_std_rad = float(direction_change_std_rad)
        self.P_min_dbm = float(tx_power_min_dbm)
        self.P_max_dbm = float(tx_power_max_dbm)
        self.sigma_noise_db = float(noise_variation_std_db)

        self._validate_parameters()

        self.PL0_db = 20.0 * np.log10(
            4.0 * np.pi * self.d0 * self.carrier_frequency_hz / self.c
        )

        # Client-specific mobility.
        self.mu_v = self._sample_uniform(mean_speed_range_mps)
        self.sigma_v = self._sample_uniform(speed_std_range_mps)
        self.rho_v = self._sample_uniform(speed_correlation_range)

        # Client-specific transmit power.
        self.mu_P_dbm = self._sample_uniform(mean_tx_power_range_dbm)
        self.sigma_P = self._sample_uniform(tx_power_std_range_db)
        self.rho_P = self._sample_uniform(tx_power_correlation_range)

        # Client/link-specific noise figure.
        self.noise_figure_db = self._sample_uniform(noise_figure_range_db)

        # Initial state.
        self.position = self._sample_position_inside_cell()
        self.direction_angle = self.rng.uniform(0.0, 2.0 * np.pi)
        self.speed = max(0.0, self.rng.normal(self.mu_v, self.sigma_v))

        self.P_dbm = float(
            np.clip(
                self.rng.normal(self.mu_P_dbm, self.sigma_P),
                self.P_min_dbm,
                self.P_max_dbm,
            )
        )

        self.X_db = self.rng.normal(0.0, self.sigma_X)
        self.h = self._complex_normal()

        self.t = 0
        self.last_displacement_m = 0.0
        self.last_shadowing_correlation = None
        self.last_fading_correlation = None
        self.last_doppler_hz = None

    def _validate_parameters(self):
        if self.cell_radius <= 0:
            raise ValueError("cell_radius must be positive.")
        if self.rb_bandwidth_hz <= 0:
            raise ValueError("rb_bandwidth_hz must be positive.")
        if self.carrier_frequency_hz <= 0:
            raise ValueError("carrier_frequency_hz must be positive.")
        if self.delta_t <= 0:
            raise ValueError("delta_t must be positive.")
        if not (0.0 < self.rate_efficiency <= 1.0):
            raise ValueError("rate_efficiency must be in (0, 1].")
        if self.d0 <= 0:
            raise ValueError("d0 must be positive.")
        if self.eta <= 0:
            raise ValueError("path_loss_exponent must be positive.")
        if self.sigma_X < 0:
            raise ValueError("shadowing_std_db cannot be negative.")
        if self.shadowing_corr_distance_m <= 0:
            raise ValueError("shadowing_corr_distance_m must be positive.")
        if self.P_min_dbm > self.P_max_dbm:
            raise ValueError("tx_power_min_dbm cannot exceed tx_power_max_dbm.")
        if self.sigma_noise_db < 0:
            raise ValueError("noise_variation_std_db cannot be negative.")

    def _sample_uniform(self, value_range):
        lo, hi = value_range
        if lo > hi:
            raise ValueError(f"Invalid range: {value_range}")
        return float(self.rng.uniform(lo, hi))

    def _sample_position_inside_cell(self):
        r = self.cell_radius * np.sqrt(self.rng.uniform(0.0, 1.0))
        theta = self.rng.uniform(0.0, 2.0 * np.pi)
        return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=float)

    def _complex_normal(self):
        return (
            self.rng.normal(0.0, 1.0 / np.sqrt(2.0))
            + 1j * self.rng.normal(0.0, 1.0 / np.sqrt(2.0))
        )

    def _update_mobility(self):
        old_position = self.position.copy()

        eps_v = self.rng.normal(0.0, self.sigma_v)
        self.speed = (
            self.rho_v * self.speed
            + (1.0 - self.rho_v) * self.mu_v
            + np.sqrt(max(0.0, 1.0 - self.rho_v**2)) * eps_v
        )
        self.speed = max(0.0, self.speed)

        self.direction_angle += self.rng.normal(0.0, self.direction_change_std_rad)
        direction = np.array([np.cos(self.direction_angle), np.sin(self.direction_angle)])

        self.position += self.speed * self.delta_t * direction

        distance_from_center = np.linalg.norm(self.position)
        if distance_from_center > self.cell_radius:
            self.position = self.position / distance_from_center * self.cell_radius
            self.direction_angle += np.pi

        self.last_displacement_m = float(np.linalg.norm(self.position - old_position))

    def _distance_to_bs(self):
        d = np.linalg.norm(self.position - self.bs_position)
        return max(float(d), self.d0)

    def _path_loss_db(self, distance):
        return self.PL0_db + 10.0 * self.eta * np.log10(distance / self.d0)

    def _update_power(self):
        eps_P = self.rng.normal(0.0, self.sigma_P)

        self.P_dbm = (
            self.rho_P * self.P_dbm
            + (1.0 - self.rho_P) * self.mu_P_dbm
            + np.sqrt(max(0.0, 1.0 - self.rho_P**2)) * eps_P
        )

        self.P_dbm = float(np.clip(self.P_dbm, self.P_min_dbm, self.P_max_dbm))

    def _update_shadowing(self):
        rho_X = np.exp(-self.last_displacement_m / self.shadowing_corr_distance_m)
        rho_X = float(np.clip(rho_X, 0.0, 0.999999))

        eps_X = self.rng.normal(0.0, self.sigma_X)
        self.X_db = rho_X * self.X_db + np.sqrt(max(0.0, 1.0 - rho_X**2)) * eps_X

        self.last_shadowing_correlation = rho_X

    def _doppler_correlation(self):
        f_D = self.speed * self.carrier_frequency_hz / self.c

        rho_h = float(bessel_j0(2.0 * np.pi * f_D * self.delta_t))
        rho_h = float(np.clip(rho_h, -0.999999, 0.999999))

        self.last_doppler_hz = float(f_D)
        self.last_fading_correlation = rho_h

        return rho_h

    def _update_fading(self):
        eps_h = self._complex_normal()
        rho_h = self._doppler_correlation()
        self.h = rho_h * self.h + np.sqrt(max(0.0, 1.0 - rho_h**2)) * eps_h

    def _noise_power_dbm(self):
        thermal_noise_dbm = self.noise_density_dbm_per_hz + 10.0 * np.log10(
            self.rb_bandwidth_hz
        )
        noise_variation = self.rng.normal(0.0, self.sigma_noise_db)

        return thermal_noise_dbm + self.noise_figure_db + noise_variation

    def __call__(self, return_details=False):
        """
        Advance the wireless state by one time step.

        Returns
        -------
        float or dict
            If return_details=False, returns communication rate in bit/s.
            If return_details=True, returns detailed physical diagnostics.
        """
        self.t += 1

        self._update_mobility()
        self._update_power()
        self._update_shadowing()
        self._update_fading()

        distance = self._distance_to_bs()
        path_loss_db = self._path_loss_db(distance)

        fading_power = np.abs(self.h) ** 2
        fading_db = 10.0 * np.log10(fading_power + 1e-12)

        noise_dbm = self._noise_power_dbm()

        snr_db = self.P_dbm - path_loss_db - self.X_db + fading_db - noise_dbm
        snr_linear = 10.0 ** (snr_db / 10.0)

        communication_rate_bps = (
            self.rate_efficiency
            * self.rb_bandwidth_hz
            * np.log2(1.0 + snr_linear)
        )

        if return_details:
            return {
                "client_id": self.client_id,
                "time_step": self.t,
                "communication_rate_bps": communication_rate_bps,
                "communication_rate_mbps": communication_rate_bps / 1e6,
                "snr_db": snr_db,
                "distance_m": distance,
                "power_dbm": self.P_dbm,
                "path_loss_db": path_loss_db,
                "shadowing_db": self.X_db,
                "shadowing_correlation": self.last_shadowing_correlation,
                "fading_power": fading_power,
                "fading_db": fading_db,
                "fading_correlation": self.last_fading_correlation,
                "doppler_hz": self.last_doppler_hz,
                "noise_dbm": noise_dbm,
                "noise_figure_db": self.noise_figure_db,
                "speed_mps": self.speed,
                "position": self.position.copy(),
                "rb_bandwidth_hz": self.rb_bandwidth_hz,
                "carrier_frequency_hz": self.carrier_frequency_hz,
                "rate_efficiency": self.rate_efficiency,
            }

        return float(communication_rate_bps)


# =============================================================================
# Physical system manager
# =============================================================================
class PhysicalSystem:
    """
    Manager for the complete physical system.

    The system creates K physical clients. Each client has one PhysicalWirelessClient

    Calling the system once advances all physical clients by one time step and
    returns communication rates for all clients
    """

    def __init__(
        self,
        num_clients,
        seed=42,
        wireless_config=None,
    ):
        if num_clients <= 0:
            raise ValueError("num_clients must be positive.")

        self.num_clients = int(num_clients)
        self.seed = int(seed)
        self.wireless_config = {} if wireless_config is None else dict(wireless_config)

        self.t = 0

        root_seed_sequence = np.random.SeedSequence(self.seed)
        wireless_client_seeds = root_seed_sequence.spawn(self.num_clients)

        self.wireless_clients = [
            PhysicalWirelessClient(
                client_id=k,
                rng=wireless_client_seeds[k],
                **self.wireless_config,
            )
            for k in range(self.num_clients)
        ]


    def __call__(self, return_details=False):
        """
        Advance the whole physical system by one time step.

        Parameters
        ----------
        return_details : bool
            If False, return only arrays of communication rates.
            If True, also return per-client wireless diagnostic dictionaries.

        Returns
        -------
        dict
            Physical state of all clients at the current time step.
        """
        self.t += 1

        communication_rates_bps = np.zeros(self.num_clients, dtype=float)

        wireless_details = []

        for k in range(self.num_clients):
            if return_details:
                wireless_info = self.wireless_clients[k](return_details=True)

                communication_rates_bps[k] = wireless_info["communication_rate_bps"]

                wireless_details.append(wireless_info)
            else:
                communication_rates_bps[k] = self.wireless_clients[k]()
        state = {
            "time_step": self.t,
            "communication_rates_bps": communication_rates_bps,
            "communication_rates_mbps": communication_rates_bps / 1e6,
        }

        if return_details:
            state["wireless_details"] = wireless_details

        return state


# =============================================================================
# Small standalone test
# =============================================================================
# if __name__ == "__main__":
#     K = 10
#     T = 20
#     seed = 102
#
#     wireless_config = {
#         "cell_radius": 500.0,
#         "rb_bandwidth_hz": 1e6,
#         "carrier_frequency_hz": 3.5e9,
#         "delta_t": 1.0,
#         "rate_efficiency": 0.65,
#         "path_loss_exponent": 3.2,
#         "shadowing_std_db": 8.0,
#         "shadowing_corr_distance_m": 30.0,
#         "mean_speed_range_mps": (0.2, 3.0),
#         "mean_tx_power_range_dbm": (15.0, 23.0),
#         "noise_figure_range_db": (5.0, 10.0),
#     }
#
#     physical_system = PhysicalSystem(
#         num_clients=K,
#         seed=seed,
#         wireless_config=wireless_config,
#     )
#
#     communication_rates_mbps = np.zeros((T, K))
#
#     for t in range(T):
#         state = physical_system(return_details=True)
#         communication_rates_mbps[t, :] = state["communication_rates_mbps"]
#
#     time_steps = np.arange(1, T + 1)
#
#     plt.figure(figsize=(10, 5))
#     for k in range(K):
#         plt.plot(time_steps, communication_rates_mbps[:, k], label=f"Client {k}")
#
#     plt.xlabel("Time Step")
#     plt.ylabel("Communication Rate (Mbps)")
#     plt.title("Time-Varying Communication Rate per Client")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()