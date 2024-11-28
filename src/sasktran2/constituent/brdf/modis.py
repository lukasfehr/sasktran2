from typing import TYPE_CHECKING

import numpy as np

from sasktran2 import Atmosphere
from sasktran2.atmosphere import (
    NativeGridDerivative,
    SurfaceDerivativeMapping,
)

from ..base import Constituent
from . import (
    MODISStokes_1,
    MODISStokes_3,
    WavelengthInterpolatorMixin,
)


class MODIS(Constituent, WavelengthInterpolatorMixin):
    def __init__(
        self,
        isotropic: np.array,
        volumetric: np.array = 0.0,
        geometric: np.array = 0.0,
        wavelengths_nm: np.array = None,
        out_of_bounds_mode="zero",
    ) -> None:
        """
        Parameters
        ----------
        isotropic : np.array
            Isotropic component (contribution weight of Lambertian).
        volumetric : np.array, optional
            Volumetric component (contribution weight of RossThick kernel), by default 0.
        geometric : np.array, optional
            Geometric component (contribution weight of LiSparse-R kernel), by default 0.
        wavelengths_nm : np.array, optional
            Wavelengths in [nm] that the parameters isotropic, volumetric, geometric is specified at, by default None indicating that these parameters are scalar.
        out_of_bounds_mode : str, optional
            One of ["extend" or "zero"], "extend" will extend the last/first value if we are
            interpolating outside the grid. "zero" will set the albedo to 0 outside of the
            grid boundaries, by default "zero"
        """
        Constituent.__init__(self)
        WavelengthInterpolatorMixin.__init__(
            self,
            wavelengths_nm=wavelengths_nm,
            wavenumbers_cminv=None,
            out_of_bounds_mode=out_of_bounds_mode,
            param_length=len(np.atleast_1d(isotropic)),
        )
        self._iso = np.atleast_1d(isotropic)
        self._vol = np.atleast_1d(volumetric)
        self._geo = np.atleast_1d(geometric)

    @property
    def isotropic(self) -> np.array:
        return self._iso

    @isotropic.setter
    def isotropic(self, iso: np.array):
        self._iso = np.atleast_1d(iso)

    @property
    def volumetric(self) -> np.array:
        return self._vol

    @volumetric.setter
    def volumetric(self, vol: np.array):
        self._vol = np.atleast_1d(vol)

    @property
    def geometric(self) -> np.array:
        return self._geo

    @geometric.setter
    def geometric(self, geo: np.array):
        self._geo = np.atleast_1d(geo)

    def add_to_atmosphere(self, atmo: Atmosphere):
        if atmo.wavelengths_nm is None:
            msg = "Atmosphere must have wavelengths defined before using MODIS"
            raise ValueError(msg)

        print(atmo.surface.brdf_args.shape)
        atmo.surface.brdf = MODISStokes_1() if atmo.nstokes == 1 else MODISStokes_3()

        interp_matrix = self._interpolating_matrix(atmo)
        print(atmo.surface.brdf_args.shape)
        print(interp_matrix.shape)
        print(self._iso.shape)
        atmo.surface.brdf_args[0, :] = interp_matrix @ self._iso
        atmo.surface.brdf_args[1, :] = interp_matrix @ self._vol
        atmo.surface.brdf_args[2, :] = interp_matrix @ self._geo
        atmo.surface.d_brdf_args[0][:, :] = 1

    def register_derivative(self, atmo: Atmosphere, name: str):
        return {}


if __name__ == "__main__":
    from sasktran2 import Config, Geometry1D, GeometryType, InterpolationMethod

    z = np.linspace(0, 65e3, 66)
    model_geometry = Geometry1D(
        cos_sza=0.5,
        solar_azimuth=0.0,
        earth_radius_m=6e6,
        altitude_grid_m=z,
        interpolation_method=InterpolationMethod.LinearInterpolation,
        geometry_type=GeometryType.Spherical,
    )
    config = Config()
    atmo = Atmosphere(
        model_geometry=model_geometry,
        config=config,
        wavelengths_nm=np.array([340.0, 440.0]),
    )
    modis = MODIS(0.2, 0.05, 0.05)
    modis.add_to_atmosphere(atmo)
    atmo.surface.brdf()