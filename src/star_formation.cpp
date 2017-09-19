/*
 * star_formation.cpp
 *
 *  Created on: 17May,2017
 *      Author: clagos
 */

#include <cmath>
#include <gsl/gsl_integration.h>

#include "star_formation.h"
#include "numerical_constants.h"

namespace shark {

struct galaxy_properties_for_integration {
	double sigma_gas0;
	double sigma_star0;
	double re;
	double rse;
};

StarFormationParameters::StarFormationParameters(const Options &options) :
	Molecular_BR_Law(0),
	nu_sf(0),
	Po(0),
	beta_press(0),
	Accuracy_SFeqs(0),
	gas_velocity_dispersion(0)
{
	options.load("star_formation.Molecular_BR_law", Molecular_BR_Law);
	options.load("star_formation.nu_sf", nu_sf);
	options.load("star_formation.Po", Po);
	options.load("star_formation.beta_press", beta_press);
	options.load("star_formation.Accuracy_SFeqs", Accuracy_SFeqs);
	options.load("star_formation.gas_velocity_dispersion", gas_velocity_dispersion);
}


StarFormation::StarFormation(StarFormationParameters parameters, std::shared_ptr<Cosmology> cosmology) :
	parameters(parameters),
	cosmology(cosmology)
{
	// no-op
}

double StarFormation::star_formation_rate(double mcold, double mstar, double rgas, double rstar, double z) {

	if (mcold <= 0 || rgas <= 0) {
		return 0;
	}

	int smax = 1000;
	gsl_integration_workspace * w
	    = gsl_integration_workspace_alloc (smax);

	/**
	 * All input quantities should be in physical units.
	 */
	// Define properties that are input for the SFR calculation.

	double re = cosmology->comoving_to_physical_size(rgas / constants::RDISK_HALF_SCALE, z);
	double rse = cosmology->comoving_to_physical_size(rstar / constants::RDISK_HALF_SCALE, z);

	double Sigma_gas = cosmology->comoving_to_physical_mass(mcold) / constants::PI2 / (re * re);
	double Sigma_star = 0;
	if(mstar){
		Sigma_star = cosmology->comoving_to_physical_mass(mstar) / constants::PI2 / (rse * rse) ;
	}

	galaxy_properties_for_integration props = {
		Sigma_gas,
		Sigma_star,
		re,
		rse
	};

	struct StarFormationAndProps {
		StarFormation *star_formation;
		galaxy_properties_for_integration *props;
	};

	StarFormationAndProps sf_and_props = {this, &props};
	auto f = [](double r, void *ctx) -> double {
		StarFormationAndProps *sf_and_props = reinterpret_cast<StarFormationAndProps *>(ctx);
		return sf_and_props->star_formation->star_formation_rate_surface_density(r, sf_and_props->props);
	};

	gsl_function F;
	F.function = f;
	F.params = &sf_and_props;

	double result, error;
	double rmin = 0;
	double rmax = 5.0*re;

	/**
	 * Here, we integrate the SFR surface density profile out to rmax.
	 */
	gsl_integration_qags (&F, rmin, rmax, 0.0, 0.02, smax,
	                        w, &result, &error);

	gsl_integration_workspace_free (w);

	// Avoid negative values.
	if(result <0){
		result = 0.0;
	}

	return cosmology->physical_to_comoving_mass(result);

}

double StarFormation::star_formation_rate_surface_density(double r, void * params){

	using namespace constants;

	// apply molecular SF law
	auto props = reinterpret_cast<galaxy_properties_for_integration *>(params);

	double Sigma_gas = props->sigma_gas0 * std::exp(-r / props->re);

	double Sigma_stars = 0;

	// Define Sigma_stars only if stellar mass and radius are positive.
	if(props->rse > 0 && props->sigma_star0 > 0){
		Sigma_stars = props->sigma_star0 * std::exp(-r / props->rse);
	}

	return PI2 * parameters.nu_sf * fmol(Sigma_gas, Sigma_stars, r) * Sigma_gas * r; //Add the 2PI*r to Sigma_SFR to make integration.
}

double StarFormation::fmol(double Sigma_gas, double Sigma_stars, double r){

	double rmol = std::pow((midplane_pressure(Sigma_gas,Sigma_stars,r)/parameters.Po),parameters.beta_press);

	return rmol/(1+rmol);
}

double StarFormation::midplane_pressure(double Sigma_gas, double Sigma_stars, double r){

	/**
	 * This function calculate the midplane pressure of the disk, and returns it in units of K/cm^-3.
	 */
	//I need to first convert all my quantities to physical quantities. Do this by calling functions from cosmology.!!!

	using namespace constants;

	double hstar = 0.14 * r; //scaleheight of the stellar disk; from Kregel et al. (2002).
	double veldisp_star = std::sqrt(PI * G * hstar * Sigma_stars); //stellar velocity dispersion in km/s.

	double star_comp = 0;
	if (Sigma_stars > 0 and veldisp_star > 0) {
		star_comp = (parameters.gas_velocity_dispersion / veldisp_star) * Sigma_stars;
	}

	double pressure = PIO2 * G_MPCGYR2 * Sigma_gas * (Sigma_gas + star_comp); //in units of Msun/Mpc/Gyr^2.

	return pressure * Pressure_SimUnits_cgs / k_Boltzmann_erg; //pressure in units of K/cm^3.
}
}  // namespace shark


