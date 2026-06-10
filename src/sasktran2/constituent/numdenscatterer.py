from __future__ import annotations

from typing import Any

import numpy as np

from sasktran2.atmosphere import Atmosphere
from sasktran2.optical.base import OpticalProperty
from sasktran2.mie.distribution import ParticleSizeDistribution
from sasktran2.util.interpolation import linear_interpolating_matrix
from sasktran2.util.state import EquationOfState
from sasktran2.constants import AVOGADRO

from .base import Constituent


class NumberDensityScatterer(Constituent):
    def __init__(
        self,
        optical_property: OpticalProperty,
        altitudes_m: np.array,
        number_density: np.array,
        out_of_bounds_mode: str = "zero",
        **kwargs,
    ) -> None:
        """
        A scattering constituent that is defined by a number density on an altitude grid and an optical property

        Parameters
        ----------
        optical_property : OpticalProperty
            The optical property defining the scattering information
        altitudes_m : np.array
            The altitude grid in [m]
        number_density : np.array
            Number density in [m^-3]
        out_of_bounds_mode : str, optional
            Interpolation mode outside of the boundaries, "extend" and "zero" are supported, by default "zero"
        kwargs : dict
            Additional arguments to pass to the optical property.
        """
        super().__init__()

        self._out_of_bounds_mode = out_of_bounds_mode
        self._altitudes_m = altitudes_m
        self._number_density = number_density
        self._optical_property = optical_property

        # Extra factor to apply to the vertical derivatives, used by the derived Extinction class
        self._vertical_deriv_factor = np.ones_like(number_density)

        # Optical derivatives can also have derivatives to this factor
        self._d_vertical_deriv_factor = {}

        self._wf_name = "number_density"

        self._kwargs = kwargs

    def __getattr__(self, __name: str) -> Any:
        if __name in self.__dict__.get("_kwargs", {}):
            return self._kwargs[__name]
        return None

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.__dict__.get("_kwargs", {}):
            self._kwargs[__name] = __value
        else:
            super().__setattr__(__name, __value)

    @property
    def number_density(self):
        return self._number_density

    @number_density.setter
    def number_density(self, number_density: np.array):
        self._number_density = number_density

    def add_to_atmosphere(self, atmo: Atmosphere):
        interp_matrix = linear_interpolating_matrix(
            self._altitudes_m,
            atmo.model_geometry.altitudes(),
            self._out_of_bounds_mode.lower(),
        )

        interped_kwargs = {k: interp_matrix @ v for k, v in self._kwargs.items()}

        self._optical_quants = self._optical_property.atmosphere_quantities(
            atmo, **interped_kwargs
        )

        interp_numden = interp_matrix @ self._number_density

        atmo.storage.total_extinction[:] += (
            self._optical_quants.extinction * (interp_numden)[:, np.newaxis]
        )

        # Optical quants in SSA temporarily stores the SSA * extinction
        atmo.storage.ssa[:] += self._optical_quants.ssa * (interp_numden)[:, np.newaxis]

        atmo.storage.leg_coeff[:] += (
            self._optical_quants.ssa[np.newaxis, :, :]
            * (interp_numden)[np.newaxis, :, np.newaxis]
            * self._optical_quants.leg_coeff
        )

        # Convert back to SSA for ease of use later in the derivatives
        self._optical_quants.ssa[:] /= self._optical_quants.extinction
        self._optical_quants.ssa[~np.isfinite(self._optical_quants.ssa)] = 1

    def register_derivative(self, atmo: Atmosphere, name: str):
        interp_matrix = linear_interpolating_matrix(
            self._altitudes_m,
            atmo.model_geometry.altitudes(),
            self._out_of_bounds_mode.lower(),
        )
        interped_kwargs = {k: interp_matrix @ v for k, v in self._kwargs.items()}

        derivs = {}

        # Factor to apply to legendre derivatives is
        # (species_ext) * (species_ssa) / (total_ext * total_ssa)

        deriv_mapping = atmo.storage.get_derivative_mapping(
            f"wf_{name}_{self._wf_name}"
        )

        deriv_mapping.d_extinction[:] += self._optical_quants.extinction
        deriv_mapping.d_ssa[:] += (
            self._optical_quants.extinction
            * (self._optical_quants.ssa - atmo.storage.ssa)
            / atmo.storage.total_extinction
        )
        deriv_mapping.d_leg_coeff[:] += (
            self._optical_quants.leg_coeff - atmo.storage.leg_coeff
        )
        deriv_mapping.scat_factor[:] += (
            self._optical_quants.ssa * self._optical_quants.extinction
        ) / (atmo.storage.ssa * atmo.storage.total_extinction)
        deriv_mapping.interpolator = (
            interp_matrix * self._vertical_deriv_factor[np.newaxis, :]
        )
        deriv_mapping.interp_dim = f"{name}_altitude"

        optical_derivs = self._optical_property.optical_derivatives(
            atmo, **interped_kwargs
        )

        for key, val in optical_derivs.items():
            deriv_mapping = atmo.storage.get_derivative_mapping(f"wf_{name}_{key}")

            deriv_mapping.d_extinction[:] += val.d_extinction
            # First, the optical property returns back d_scattering extinction in the d_ssa container,
            # convert this to d_ssa
            deriv_mapping.d_ssa[:] += (
                val.d_ssa - val.d_extinction * self._optical_quants.ssa
            ) / self._optical_quants.extinction
            deriv_mapping.d_leg_coeff[:] += val.d_leg_coeff

            if key in self._d_vertical_deriv_factor:
                # Have to make some adjustments

                # The change in extinction is adjusted
                deriv_mapping.d_extinction[:] += (
                    self._optical_quants.extinction
                    / (interp_matrix @ self._vertical_deriv_factor)[:, np.newaxis]
                    * (interp_matrix @ self._d_vertical_deriv_factor[key])[
                        :, np.newaxis
                    ]
                )

                # Change in single scatter albedo should be invariant whether or not we are
                # in extinction space or number density space

            # Start with leg_coeff
            deriv_mapping.d_leg_coeff[:] += (
                self._optical_quants.leg_coeff - atmo.storage.leg_coeff
            ) * (
                1 / self._optical_quants.ssa * deriv_mapping.d_ssa
                + 1 / self._optical_quants.extinction * deriv_mapping.d_extinction
            )[np.newaxis, :, :]

            # Then adjust d_ssa
            deriv_mapping.d_ssa[:] *= self._optical_quants.extinction
            deriv_mapping.d_ssa[:] += deriv_mapping.d_extinction * (
                self._optical_quants.ssa - atmo.storage.ssa
            )
            deriv_mapping.d_ssa[:] /= atmo.storage.total_extinction

            # TODO: The model should probably handle this
            norm_factor = deriv_mapping.d_leg_coeff.max(axis=0)
            norm_factor[norm_factor == 0] = 1

            deriv_mapping.scat_factor[:] = (
                self._optical_quants.ssa * self._optical_quants.extinction
            ) / (atmo.storage.ssa * atmo.storage.total_extinction)

            deriv_mapping.d_leg_coeff[:] /= norm_factor[np.newaxis, :, :]
            deriv_mapping.scat_factor[:] *= norm_factor

            deriv_mapping.interpolator = (
                interp_matrix * self._number_density[np.newaxis, :]
            )
            deriv_mapping.interp_dim = f"{name}_altitude"

        return derivs


class ExtinctionScatterer(NumberDensityScatterer):
    def __init__(
        self,
        optical_property: OpticalProperty,
        altitudes_m: np.array,
        extinction_per_m: np.array,
        extinction_wavelength_nm: float,
        out_of_bounds_mode: str = "zero",
        **kwargs,
    ) -> None:
        """
        A scattering constituent that is defined by a number density on an altitude grid and an optical property

        Parameters
        ----------
        optical_property : OpticalProperty
            The optical property defining the scattering information
        altitudes_m : np.array
            The altitude grid in [m]
        extinction_per_m : np.array
            Extinction in [m^-1]
        extinction_wavelength_nm : float
            Wavelength that the extinction profile is specified at
        out_of_bounds_mode : str, optional
            Interpolation mode outside of the boundaries, "extend" and "zero" are supported, by default "zero"
        kwargs : dict
            Additional arguments passed to the optical property
        """
        self._extinction_per_m = extinction_per_m
        self._extinction_wavelength_nm = extinction_wavelength_nm

        super().__init__(
            optical_property, altitudes_m, None, out_of_bounds_mode, **kwargs
        )
        self._extinction_to_numden_factors = None
        self._update_numberdensity()
        self._wf_name = "extinction"

    def _update_numberdensity(self):
        self._extinction_to_numden_factors = self._optical_property.cross_sections(
            np.array([self._extinction_wavelength_nm]),
            altitudes_m=self._altitudes_m,
            **self._kwargs,
        ).extinction.flatten()
        self._vertical_deriv_factor = 1 / self._extinction_to_numden_factors

        self.number_density = (
            self._extinction_per_m / self._extinction_to_numden_factors
        )

        self._d_vertical_deriv_factor = (
            self._optical_property.cross_section_derivatives(
                np.array([self._extinction_wavelength_nm]),
                altitudes_m=self._altitudes_m,
                **self._kwargs,
            )
        )

        for _, val in self._d_vertical_deriv_factor.items():
            # convert from derivative of x to derivative of 1/x
            val *= -1 * self._vertical_deriv_factor**2  # noqa: PLW2901

    @property
    def extinction_per_m(self):
        return self._extinction_per_m

    @extinction_per_m.setter
    def extinction_per_m(self, extinction: np.array):
        self._extinction_per_m = extinction

    def add_to_atmosphere(self, atmo: Atmosphere):
        self._update_numberdensity()
        super().add_to_atmosphere(atmo)


class MassMixingRatioScatterer(Constituent):
    def __init__(
        self,
        optical_property: OpticalProperty,
        altitudes_m: np.array,
        mass_mixing_ratio: np.array,
        density: np.array,
        distribution: ParticleSizeDistribution,
        size_conversion: float = 1e-9,
        out_of_bounds_mode: str = "zero",
        **kwargs,
    ) -> None:
        """
        A scattering constituent that is defined by a number density on an altitude grid and an optical property

        Parameters
        ----------
        optical_property : OpticalProperty
            The optical property defining the scattering information
        altitudes_m : np.array
            The altitude grid in [m]
        mass_mixing_ratio : np.array
            Mass mixing ratio in [kg/kg]
        density : float
            Particle density in [kg/m3]
        distribution : ParticleSizeDistribution
            Size distribution corresponding to optical_property, used to determine average particle mass.
        size_conversion : float, optional
            Factor required to convert units of distribution to m, by default 1e-9 (nm).
        out_of_bounds_mode : str, optional
            Interpolation mode outside of the boundaries, "extend" and "zero" are supported, by default "zero"
        kwargs : dict
            Additional arguments passed to the optical property
        """
        super().__init__()

        self._optical_property = optical_property
        self._altitudes_m = altitudes_m
        self._mmr = mass_mixing_ratio
        self._density = density
        self._distribution = distribution
        self._size_conversion = size_conversion
        self._out_of_bounds_mode = out_of_bounds_mode
        self._kwargs = kwargs

    def __getattr__(self, __name: str) -> Any:
        if __name in self.__dict__.get("_kwargs", {}):
            return self._kwargs[__name]
        return None

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.__dict__.get("_kwargs", {}):
            self._kwargs[__name] = __value
        else:
            super().__setattr__(__name, __value)

    @property
    def mass_mixing_ratio(self):
        return self._mmr

    @mass_mixing_ratio.setter
    def mass_mixing_ratio(self, mass_mixing_ratio: np.array):
        self._mmr = mass_mixing_ratio

    def _mean_particle_volume(self, **kwargs):
        # get average particle masses and any derivatives

        c = self._size_conversion**3 * 4.0 * np.pi / 3.0
        keys = [key for key in kwargs if key in self._distribution.args()]
        if keys:
            n = len(kwargs[keys[0]])
            v = np.zeros(n)
            dv_dx = {key: np.zeros(n) for key in keys}
            for i in range(n):
                # could cache identical key combinations to speed this up
                kw = {key: kwargs[key][i] for key in keys}
                rv = self._distribution.distribution(**kw)
                v[i] = c * rv.moment(order=3)
                for key in keys:
                    dkey = max(1e-6, 1e-6 * kw[key])
                    kw[key] += dkey
                    rv = self._distribution.distribution(**kw)
                    dv_dx[key][i] = (c * rv.moment(order=3) - v[i]) / dkey
        else:
            rv = self._distribution.distribution()
            v = {"V": c * rv.moment(order=3)}
            dv_dx = {}
        return v, dv_dx

    def add_to_atmosphere(self, atmo: Atmosphere):
        interp_matrix = linear_interpolating_matrix(
            self._altitudes_m,
            atmo.model_geometry.altitudes(),
            self._out_of_bounds_mode.lower(),
        )

        interped_kwargs = {k: interp_matrix @ v for k, v in self._kwargs.items()}
        self._optical_quants = self._optical_property.atmosphere_quantities(
            atmo, **interped_kwargs
        )

        dry_air_number_density = atmo.state_equation.dry_air_numberdensity
        self._volume, self._d_volume = self._mean_particle_volume(**interped_kwargs)

        interp_mmr = interp_matrix @ self._mmr
        self._u_to_n_factor = (  # mass mixing ratio to number density
            atmo.state_equation._M
            * dry_air_number_density["N"]
            / (self._density * self._volume * AVOGADRO)
        )
        self._number_density = interp_mmr * self._u_to_n_factor

        atmo.storage.total_extinction[:] += (
            self._optical_quants.extinction * self._number_density[:, np.newaxis]
        )

        # Optical quants in SSA temporarily stores the SSA * extinction
        atmo.storage.ssa[:] += (
            self._optical_quants.ssa * self._number_density[:, np.newaxis]
        )

        atmo.storage.leg_coeff[:] += (
            self._optical_quants.ssa[np.newaxis, :, :]
            * self._number_density[np.newaxis, :, np.newaxis]
            * self._optical_quants.leg_coeff
        )

        # Convert back to SSA for ease of use later in the derivatives
        self._optical_quants.ssa[:] /= self._optical_quants.extinction
        self._optical_quants.ssa[~np.isfinite(self._optical_quants.ssa)] = 1

    def register_derivative(self, atmo, name):
        dry_air_number_density = atmo.state_equation.dry_air_numberdensity

        interp_matrix = linear_interpolating_matrix(
            self._altitudes_m,
            atmo.model_geometry.altitudes(),
            self._out_of_bounds_mode.lower(),
        )

        # derivatives wrt mass mixing ratio
        deriv_mapping = atmo.storage.get_derivative_mapping(
            f"wf_{name}_mass_mixing_ratio"
        )
        deriv_mapping.d_extinction[:] += (
            self._optical_quants.extinction * self._u_to_n_factor[:, np.newaxis]
        )
        deriv_mapping.d_ssa[:] += (
            self._optical_quants.extinction
            * self._u_to_n_factor[:, np.newaxis]
            * (self._optical_quants.ssa - atmo.storage.ssa)
            / atmo.storage.total_extinction
        )
        deriv_mapping.d_leg_coeff[:] += (
            self._optical_quants.leg_coeff - atmo.storage.leg_coeff
        )
        deriv_mapping.scat_factor[:] += (
            self._optical_quants.ssa
            * self._optical_quants.extinction
            * self._u_to_n_factor[:, np.newaxis]
        ) / (atmo.storage.ssa * atmo.storage.total_extinction)
        deriv_mapping.interpolator = interp_matrix
        deriv_mapping.interp_dim = f"{name}_altitude"

        # contributions from the change in number density due to a constant MMR and changing pressure/temperature
        deriv_names = []
        d_vals = []
        if atmo.calculate_pressure_derivative:
            deriv_names.append("pressure_pa")
            d_vals.append(dry_air_number_density["dN_dP"])
        if atmo.calculate_temperature_derivative:
            deriv_names.append("temperature_k")
            d_vals.append(dry_air_number_density["dN_dT"])
        if atmo.calculate_specific_humidity_derivative:
            deriv_names.append("specific_humidity")
            d_vals.append(dry_air_number_density["dN_dsh"])

        for deriv_name, vert_factor in zip(deriv_names, d_vals, strict=False):
            deriv_mapping = atmo.storage.get_derivative_mapping(
                f"wf_{name}_{deriv_name}"
            )
            deriv_mapping.d_extinction[:] += self._optical_quants.extinction
            deriv_mapping.d_ssa[:] += (
                self._optical_quants.extinction
                * (self._optical_quants.ssa - atmo.storage.ssa)
                / atmo.storage.total_extinction
            )
            deriv_mapping.d_leg_coeff[:] += (
                self._optical_quants.leg_coeff - atmo.storage.leg_coeff
            )
            deriv_mapping.scat_factor[:] += (
                self._optical_quants.ssa
                * self._optical_quants.extinction
                / (atmo.storage.ssa * atmo.storage.total_extinction)
            )
            deriv_mapping.interpolator = (
                np.eye(len(self._number_density))
                * (vert_factor * self._number_density / dry_air_number_density["N"])[
                    np.newaxis, :
                ]
            )
            deriv_mapping.interp_dim = "altitude"
            deriv_mapping.assign_name = f"wf_{deriv_name}"

        # contributions from derivatives of the optical property
        interped_kwargs = {k: interp_matrix @ v for k, v in self._kwargs.items()}
        optical_derivs = self._optical_property.optical_derivatives(
            atmo, **interped_kwargs
        )

        for key, val in optical_derivs.items():
            deriv_mapping = atmo.storage.get_derivative_mapping(f"wf_{name}_{key}_xs")
            deriv_mapping.d_extinction[:] += val.d_extinction
            deriv_mapping.d_ssa[:] += (
                val.d_ssa - atmo.storage.ssa * val.d_extinction
            ) / atmo.storage.total_extinction
            deriv_mapping.d_leg_coeff[:] += val.d_leg_coeff + (
                self._optical_quants.leg_coeff - atmo.storage.leg_coeff
            ) * (
                val.d_ssa / (self._optical_quants.ssa * self._optical_quants.extinction)
            )
            deriv_mapping.scat_factor[:] += (
                self._optical_quants.ssa
                * self._optical_quants.extinction
                / (atmo.storage.ssa * atmo.storage.total_extinction)
            )

            # TODO: The model should probably handle this
            norm_factor = deriv_mapping.d_leg_coeff.max(axis=0)
            norm_factor /= 10  # this seems to improve agreement with numeric
            norm_factor[norm_factor == 0] = 1
            deriv_mapping.d_leg_coeff[:] /= norm_factor[np.newaxis, :, :]
            deriv_mapping.scat_factor[:] *= norm_factor

            deriv_mapping.interpolator = (
                interp_matrix * self._number_density[:, np.newaxis]
            )
            deriv_mapping.interp_dim = f"{name}_altitude"
            deriv_mapping.assign_name = f"wf_{name}_{key}"
            # might want to adjust the interpolator here if the optical_derivs include pressure/temperature/humidity

        # contributions from derivatives of average particle size
        for deriv_name, vert_factor in self._d_volume.items():
            deriv_mapping = atmo.storage.get_derivative_mapping(
                f"wf_{name}_{deriv_name}_particle_size"
            )
            deriv_mapping.d_extinction[:] += self._optical_quants.extinction
            deriv_mapping.d_ssa[:] += (
                self._optical_quants.extinction
                * (self._optical_quants.ssa - atmo.storage.ssa)
                / atmo.storage.total_extinction
            )
            deriv_mapping.d_leg_coeff[:] += (
                self._optical_quants.leg_coeff - atmo.storage.leg_coeff
            )
            deriv_mapping.scat_factor[:] += (
                self._optical_quants.ssa
                * self._optical_quants.extinction
                / (atmo.storage.ssa * atmo.storage.total_extinction)
            )

            # TODO: The model should probably handle this
            norm_factor = deriv_mapping.d_leg_coeff.max(axis=0)
            norm_factor /= 10  # this seems to improve agreement with numeric
            norm_factor[norm_factor == 0] = 1
            deriv_mapping.d_leg_coeff[:] /= norm_factor[np.newaxis, :, :]
            deriv_mapping.scat_factor[:] *= norm_factor

            deriv_mapping.interpolator = (
                interp_matrix
                * (-vert_factor / self._volume * self._number_density)[:, np.newaxis]
            )
            deriv_mapping.interp_dim = f"{name}_altitude"
            deriv_mapping.assign_name = f"wf_{name}_{deriv_name}"
