//
// ICRAR - International Centre for Radio Astronomy Research
// (c) UWA - The University of Western Australia, 2017
// Copyright by UWA (in the framework of the ICRAR)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//

/**
 * @file
 */

#include <cmath>
#include <random>
#include <vector>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

#include "agn_feedback.h"
#include "components.h"
#include "galaxy_mergers.h"
#include "numerical_constants.h"
#include "physical_model.h"

namespace shark {

GalaxyMergerParameters::GalaxyMergerParameters(const Options &options)
{
	options.load("galaxy_mergers.major_merger_ratio", major_merger_ratio, true);
	options.load("galaxy_mergers.minor_merger_burst_ratio", minor_merger_burst_ratio, true);
	options.load("galaxy_mergers.gas_fraction_burst_ratio", gas_fraction_burst_ratio, true);

	options.load("galaxy_mergers.merger_random_seed", merger_random_seed);

	options.load("galaxy_mergers.jiang08_a", jiang08[0], true);
	options.load("galaxy_mergers.jiang08_b", jiang08[1], true);
	options.load("galaxy_mergers.jiang08_c", jiang08[2], true);
	options.load("galaxy_mergers.jiang08_d", jiang08[3], true);

	options.load("galaxy_mergers.tau_delay", tau_delay);
	options.load("galaxy_mergers.mass_min", mass_min);

	options.load("galaxy_mergers.f_orbit", f_orbit);
	options.load("galaxy_mergers.cgal", cgal);
	options.load("galaxy_mergers.fgas_dissipation", fgas_dissipation);
	options.load("galaxy_mergers.merger_ratio_dissipation", merger_ratio_dissipation);
}

GalaxyMergers::GalaxyMergers(GalaxyMergerParameters parameters,
		const CosmologyPtr &cosmology,
		SimulationParameters simparams,
		const DarkMatterHalosPtr &darkmatterhalo,
		std::shared_ptr<BasicPhysicalModel> physicalmodel,
		const AGNFeedbackPtr &agnfeedback) :
	parameters(parameters),
	cosmology(cosmology),
	simparams(simparams),
	darkmatterhalo(darkmatterhalo),
	physicalmodel(physicalmodel),
	agnfeedback(agnfeedback),
	generator(),
	distribution(-0.14, 0.26)
{
	// no-op
}


void GalaxyMergers::orbital_parameters(double &vr, double &vt, double f){

	//double f2 = 1+1/std::max(mass_ratio,1+tolerance);

	//Now generate two random numbers between 0 and 3.
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0, 3);

	vr = distribution(generator);
	vt = distribution(generator);
}

double GalaxyMergers::merging_timescale_orbital(){

	/**
	 * Uses function calculated in Lacey & Cole (1993), who found that it was best described by a log
	 * normal distribution with median value -0.14 and dispersion 0.26.
	 */

	//TODO: add other dynamical friction timescales.

	return distribution(generator);

}

double GalaxyMergers::mass_ratio_function(double mp, double ms){

	/**
	 * Input variables:
	 * mp: mass  of primary galaxy.
	 * ms: mass of secondary galaxy.
	 */

	return 1+1/std::max(ms / mp, 1. + constants::tolerance);
}

double GalaxyMergers::merging_timescale_mass(double mp, double ms){

	/**
	 * Input variables:
	 * mp: mass  of primary galaxy.
	 * ms: mass of secondary galaxy.
	 */

	double mass_ratio = mp/ms;

	return 0.3722 * mass_ratio/std::log(1+mass_ratio);
}

void GalaxyMergers::merging_timescale(SubhaloPtr &primary, SubhaloPtr &secondary, double z, bool transfer_types2)
{
	auto satellites = secondary->galaxies;
	if(transfer_types2){
		satellites = secondary->all_type2_galaxies();
	}

	auto halo = primary->host_halo;

	double tau_dyn = darkmatterhalo->halo_dynamical_time(halo, z);

	double mp = primary->Mvir + primary->central_galaxy()->baryon_mass();

	for (auto &galaxy: satellites){

		// Define merging timescale and redefine type of galaxy.
		if(parameters.tau_delay > 0){
			double mgal = galaxy->baryon_mass();
			double ms = secondary->Mvir + mgal;
			if(transfer_types2){
				ms = galaxy->msubhalo_type2 + mgal;
			}
			double tau_mass = merging_timescale_mass(mp, ms);
			double tau_orbits = merging_timescale_orbital();

			galaxy->tmerge = parameters.tau_delay * tau_mass * tau_orbits* tau_dyn;
		}
		else{
			galaxy->tmerge = parameters.tau_delay;
		}

		//Only define the following parameters if the galaxies were not type=2.
		if(!transfer_types2){
			galaxy->concentration_type2 = secondary->concentration;
			galaxy->msubhalo_type2 = secondary->Mvir;
			galaxy->lambda_type2 = secondary->lambda;
		}
	}

}

void GalaxyMergers::merging_subhalos(HaloPtr &halo, double z)
{
	auto central_subhalo = halo->central_subhalo;

	if (!central_subhalo) {
		std::ostringstream os;
		os << halo << " has no central subhalo - in merging_subhalos";
		throw exception(os.str());
	}

	if(!central_subhalo->central_galaxy()){
		std::ostringstream os;
		os << "Central subhalo " << central_subhalo << " does not have central galaxy - in merging_subhalos.";
		throw invalid_argument(os.str());
	}

	for(auto &satellite_subhalo: halo->satellite_subhalos) {

		//Identify which subhalos will disappear in the next snapshot
		if (satellite_subhalo->last_snapshot_identified == satellite_subhalo->snapshot) {

			LOG(debug) << "Merging satellite subhalo " << satellite_subhalo
			           << " into central subhalo " << central_subhalo
			           << " because this is its last snapshot";

			//Calculate dynamical friction timescale for all galaxies in satellite_subhalo.
			merging_timescale(central_subhalo, satellite_subhalo, z, false);

			// Change type of galaxies to type=2 before transferring them to the central_subhalo.
			for (auto &galaxy: satellite_subhalo->galaxies){
				galaxy->galaxy_type = Galaxy::TYPE2;
			}

			//transfer all mass from the satellite_subhalo to the central_subhalo. Note that this implies a horizontal transfer of information.
			transfer_baryon_mass(central_subhalo, satellite_subhalo);

			//Now transfer the galaxies in this subhalo to the central subhalo. Note that this implies a horizontal transfer of information.
			satellite_subhalo->transfer_galaxies_to(central_subhalo);
		}
		else {
			//In cases where the subhalo does not disappear, we search for type=2 galaxies and transfer them to the central subhalo,
			//recalculating its merging timescale.

			merging_timescale(central_subhalo, satellite_subhalo, z, true);
			//Now transfer the galaxies in this subhalo to the central subhalo. Note that this implies a horizontal transfer of information.
			satellite_subhalo->transfer_type2galaxies_to(central_subhalo);
		}

		satellite_subhalo->check_satellite_subhalo_galaxy_composition();
	}

	//Now evaluate cases where central subhalo disappears in the next snapshot.
	/**
	 * In this case we do not transfer galaxies or baryonic mass from one subhalo to another
	 * as that would imply a transfer across time, which is what we later do in evolve_halos.cpp
	 */
	if(central_subhalo->last_snapshot_identified == central_subhalo->snapshot){

		// Identify the central subhalo of the halo in which the descendant subhalo lives. This is the subhalo to which
		// the galaxies should be transferred to (rather than the descendant subhalo, which by chance could be a satellite subhalo).
		auto desc_subhalo = central_subhalo->descendant->host_halo->central_subhalo;

		LOG(debug) << "Merging central subhalo " << central_subhalo
		           << " into " << desc_subhalo << " (the central subhalo of its descendant halo)"
		           << " because this is its last snapshot";

		// Find main progenitor subhalo of the descendant subhalo and use that to calculate merging timescales.
		auto primary_subhalo = desc_subhalo->main();

		// Detect cases where there is no central galaxy in the main subhalo that will merge with this one in the next snapshot.
		if(!primary_subhalo->central_galaxy()){
			std::ostringstream os;
			os << "Primary subhalo " << primary_subhalo << " (last_snapshot=";
			os << primary_subhalo->last_snapshot_identified << ") does not have central galaxy - in merging_subhalos. ";
			os << " Ascendants are: ";
			auto ascendants = primary_subhalo->ascendants;
			std::copy(ascendants.begin(), ascendants.end(), std::ostream_iterator<SubhaloPtr>(os, " "));
			os << ". Galaxies are: ";
			auto galaxies = primary_subhalo->galaxies;
			std::copy(galaxies.begin(), galaxies.end(), std::ostream_iterator<GalaxyPtr>(os, " "));
			throw invalid_argument(os.str());
		}

		//Calculate dynamical friction timescale for all galaxies disappearing in the primary subhalo of the merger in the next snapshot.
		merging_timescale(primary_subhalo, central_subhalo, z, false);

	}

}

void GalaxyMergers::merging_galaxies(HaloPtr &halo, int snapshot, double delta_t){

	/**
	 * This function determines which galaxies are merging in this snapshot by comparing tmerge with the duration of the snapshot.
	 * Inputs:
	 * halo: halo in which the two galaxies merging live.
	 * z: current redshift.
	 * delta_t: time interval between the current snapshots.
	 */

	//First define central subhalo.

	double z = simparams.redshifts[snapshot];

	auto &central_subhalo = halo->central_subhalo;

	if (!central_subhalo) {
		std::ostringstream os;
		os << halo << " has no central subhalo - in merging_galaxies";
		throw exception(os.str());
	}

	/**
	 * First find central galaxy of central subhalo.
	 */
	GalaxyPtr central_galaxy = central_subhalo->central_galaxy();
	if(!central_galaxy){
		std::ostringstream os;
		os << central_subhalo << " has no central galaxy - in merging_galaxies. Number of galaxies " << central_subhalo->galaxy_count() << ".\n";
		os << central_subhalo << " has a descendant " << central_subhalo->descendant << "which is of type " << central_subhalo->descendant->subhalo_type << "\n";
		os << central_subhalo << " has " << central_subhalo->ascendants.size() << " ascendants.\n";
		os << central_subhalo << " has a halo with " << central_subhalo->host_halo->ascendants.size() << " ascendants.";
		throw exception(os.str());
	}

	std::vector<GalaxyPtr> all_sats_to_delete;

	for (auto &galaxy: central_subhalo->galaxies){
		if(galaxy->galaxy_type == Galaxy::TYPE2){
			/**
			 * If merging timescale is less than the duration of this snapshot, then proceed to merge. Otherwise, update merging timescale.
			 */
			if(galaxy->tmerge < delta_t){
				create_merger(central_galaxy, galaxy, halo, snapshot);
				// Accumulate all satellites that we need to delete at the end.
				all_sats_to_delete.push_back(galaxy);
			}
			else{
				//check if this galaxy will merge on the next snapshot instead, and if so, redefine their descendant_id.
				if(snapshot+2 < simparams.max_snapshot){
					double z1 = simparams.redshifts[snapshot];
					double z2 = simparams.redshifts[snapshot+2];
					double delta_t_twosnaps = cosmology->convert_redshift_to_age(z2) - cosmology->convert_redshift_to_age(z1);
					if(galaxy->tmerge < delta_t_twosnaps){
						galaxy->descendant_id = central_galaxy->id;
					}
				}
				galaxy->tmerge = galaxy->tmerge - delta_t;
			}
		}
	}

	// Now destroy and remove satellite galaxy.
	central_subhalo->remove_galaxies(all_sats_to_delete);

	//calculate specific angular momentum of bulge and disk.
	//darkmatterhalo->disk_sAM(*central_subhalo , *central_galaxy);

	// Trigger starbursts in all the galaxies that have gas in the bulge.
	create_starbursts(halo, z, delta_t);

}

void GalaxyMergers::create_merger(GalaxyPtr &central, GalaxyPtr &satellite, HaloPtr &halo, int snapshot){

	/**
	 * This function classifies the merger and transfer all the baryon masses to the right component of the central.
	 * Inputs:
	 * central: central galaxy.
	 * satellite: satellite galaxy.
	 * halo: halo in which the two galaxies merging live.
	 * z: current redshift.
	 * delta_t: time interval between the current snapshots.
	 */


	//First define central subhalo.

	double mbar_central = central->baryon_mass();

	double mbar_satellite = satellite->baryon_mass();

	//Create merger only if the galaxies have a baryon_mass > 0.
	if(mbar_central <= 0 and mbar_satellite <=0 ){
		return;
	}

	double mass_ratio = mbar_satellite/mbar_central;

	//If mass ratio>1 is because satellite galaxy is more massive than central, so redefine mass ratio accordingly.
	if(mass_ratio > 1){
		mass_ratio = 1 / mass_ratio;
	}

	// define gas mass ratio of the merger.
	double ms = central->stellar_mass() + satellite->stellar_mass();
	double mgas_ratio  = 1;
	if(ms > 0){
		mgas_ratio = (central->gas_mass() + satellite->gas_mass()) / ms;
	}

	/**
	 * First, calculate remnant galaxy's bulge size based on merger properties. Assume both stars and gas
	 * settle in the same configuration.
	 */
	central->bulge_gas.rscale   = bulge_size_merger(mass_ratio, mgas_ratio, central, satellite, halo);
	central->bulge_stars.rscale = central->bulge_gas.rscale;


	// Black holes merge regardless of the merger type.
	central->smbh += satellite->smbh;

	//satellite stellar mass is always transferred to the bulge.
	transfer_history_satellite_to_bulge(central, satellite, snapshot);

	/**
	 * Depending on the mass ratio, the baryonic components of the satellite and the disk of the major galaxy are going to be transferred differently.
	 */

	// Major mergers.
	if(mass_ratio >= parameters.major_merger_ratio){

		/**
		 * Transfer mass. In the case of major mergers, transfer all stars and gas to the bulge.
		 */

		central->bulge_stars.mass += central->disk_stars.mass + satellite->stellar_mass();
		central->bulge_stars.mass_metals += central->disk_stars.mass_metals + satellite->stellar_mass_metals();

		// Keep track of stars being transfered to the bulge via assembly.
		central->galaxymergers_assembly_stars.mass += central->disk_stars.mass + satellite->stellar_mass();
		central->galaxymergers_assembly_stars.mass_metals += central->disk_stars.mass_metals + satellite->stellar_mass_metals();

		central->bulge_gas.mass += central->disk_gas.mass + satellite->gas_mass();
		central->bulge_gas.mass_metals +=  central->disk_gas.mass_metals + satellite->gas_mass_metals();

		transfer_history_disk_to_bulge(central, snapshot);

		//Make all disk values 0.
		central->disk_stars.restore_baryon();
		central->disk_gas.restore_baryon();

	}
	else{//minor mergers

		/**
		 * Transfer mass. In the case of minor mergers, transfer the satellite' stars to the central bulge, and the satellite' gas to the central's disk.
		 */

		double mgas_old_central = central->disk_gas.mass;

		// Transfer mass to bulge.
		central->bulge_stars.mass += satellite->stellar_mass();
		central->bulge_stars.mass_metals += satellite->stellar_mass_metals();

		// Keep track of stars being transfered to the bulge via assembly.
		central->galaxymergers_assembly_stars.mass += satellite->stellar_mass();
		central->galaxymergers_assembly_stars.mass_metals += satellite->stellar_mass_metals();

		// Calculate new angular momentum by adding up the two galaxies.
		double tot_am = satellite->disk_gas.angular_momentum() + satellite->bulge_gas.angular_momentum();
		double new_disk_sAM = 0;
		if(tot_am > 0){
			new_disk_sAM = (central->disk_gas.angular_momentum() + tot_am) / (central->disk_gas.mass + satellite->gas_mass());
		}
		// Modify specific AM and size based on new values.
		if(tot_am > 0){
			central->disk_gas.sAM = new_disk_sAM;
			central->disk_gas.rscale =  central->disk_gas.sAM / (2.0 * central->vmax) * constants::RDISK_HALF_SCALE;

			if (std::isnan(central->disk_gas.sAM) or std::isnan(central->disk_gas.rscale)) {
				throw invalid_argument("rgas or sAM are NaN, cannot continue at galaxy mergers - in create_merger gas-rich minor merger");
			}
		}

		// Transfer gas mass to central disk.
		central->disk_gas.mass += satellite->gas_mass();
		central->disk_gas.mass_metals +=  satellite->gas_mass_metals();

		if(mass_ratio >= parameters.minor_merger_burst_ratio and mgas_ratio > parameters.gas_fraction_burst_ratio){

			central->bulge_gas += central->disk_gas;

			//Make gas disk values 0.
			central->disk_gas.restore_baryon();
		}
		else{
			//Check cases where there is no disk in the central but the satellite is bringing gas.
			if(satellite->gas_mass() > 0 and mgas_old_central <= 0){
				double tot_am = satellite->disk_gas.angular_momentum() + satellite->bulge_gas.angular_momentum();
				central->disk_gas.sAM = tot_am / satellite->gas_mass();
				central->disk_gas.rscale = central->disk_gas.sAM / (2.0 * central->vmax) * constants::RDISK_HALF_SCALE;

				if (std::isnan(central->disk_gas.sAM) or std::isnan(central->disk_gas.rscale)) {
					throw invalid_argument("rgas or sAM are NaN, cannot continue at galaxy mergers - in create_merger gas-poor minor merger");
				}
			}
		}
	}

	//Assume both stars and gas mix up well during mergers.
	if(central->bulge_mass() > 0){
		double v_pseudo = std::sqrt(constants::G * central->bulge_mass() / central->bulge_gas.rscale);
		central->bulge_gas.sAM   = central->bulge_gas.rscale * v_pseudo;
		central->bulge_stars.sAM = central->bulge_gas.sAM;
	}

	// Calculate specific angular momentum after mass was transferred to the bulge.
	auto subhalo = halo->central_subhalo;

	if(std::isnan(central->bulge_stars.mass)){
		std::ostringstream os;
		os << central << " has a bulge mass not well defined";
		throw invalid_data(os.str());
	}

}

void GalaxyMergers::create_starbursts(HaloPtr &halo, double z, double delta_t){

	for (auto &subhalo: halo->all_subhalos()){
		for (auto &galaxy: subhalo->galaxies){
			// Trigger starburst only in case there is gas in the bulge.
			if(galaxy->bulge_gas.mass > parameters.mass_min){

				// Calculate black hole growth due to starburst.
				double delta_mbh = agnfeedback->smbh_growth_starburst(galaxy->bulge_gas.mass, subhalo->Vvir);
				double delta_mzbh = 0;

				if(galaxy->bulge_gas.mass > 0){
					delta_mzbh = delta_mbh/galaxy->bulge_gas.mass * galaxy->bulge_gas.mass_metals;
				}

				double tdyn = agnfeedback->smbh_accretion_timescale(*galaxy, z);

				// Define accretion rate.
				galaxy->smbh.macc_sb = delta_mbh/tdyn;

				// Grow SMBH.
				galaxy->smbh.mass += delta_mbh;
				galaxy->smbh.mass_metals += delta_mzbh;

				// Reduce gas available for star formation due to black hole growth.
				galaxy->bulge_gas.mass -= delta_mbh;
				galaxy->bulge_gas.mass_metals -= delta_mzbh;

				// Trigger starburst.
				physicalmodel->evolve_galaxy_starburst(*subhalo, *galaxy, z, delta_t, true);

				// Check for small gas reservoirs left in the bulge, in case mass is small, transfer to disk.
				if(galaxy->bulge_gas.mass > 0 and galaxy->bulge_gas.mass < parameters.mass_min){
					transfer_bulge_gas(subhalo, galaxy, z);
				}
			}
			else if (galaxy->bulge_gas.mass > 0){
				transfer_bulge_gas(subhalo, galaxy, z);
			}
		}
	}

}

double GalaxyMergers::bulge_size_merger(double mass_ratio, double mgas_ratio, GalaxyPtr &central, GalaxyPtr &satellite, HaloPtr &halo){

	/**
	 * This function calculates the bulge sizes resulting from a galaxy mergers following Cole et al. (2000). This assumes
	 * that the internal energy of the remnant spheroid just after the mergers is equal to the sum of the internal and relative
	 * orbital energies of the two merging galaxies (neglecting any energy dissipation and mass loss during the merger).
	 * Inputs:
	 * central: central galaxy.
	 * satellite: satellite galaxy.
	 * halo: halo in which the two galaxies merging live.
	 */

	double mtotal_central = 0;
	double rcentral = 0;
    double mbar_central = 0;
    double enc_mass = 0;

	double mbar_satellite = satellite->baryon_mass();

	double rsatellite = satellite->composite_size();

	// Define central properties depending on whether merger is major or minor merger.
	if(mass_ratio >= parameters.major_merger_ratio){

 		mbar_central = central->baryon_mass();

 		rcentral = central->composite_size();

 		auto subhalo_central = halo->central_subhalo;

 		enc_mass = darkmatterhalo->enclosed_mass(rcentral/darkmatterhalo->halo_virial_radius(*subhalo_central), subhalo_central->concentration);

 		// Because central part of the DM halo behaves like the baryons, the mass of the central galaxy includes
 		// the DM mass enclosed by rcentral.
 		mtotal_central = mbar_central + 2.0 * halo->Mvir * enc_mass;
	}
	else{
		// In this case use the same equations as in major mergers, but changing the total mass of the central that will end up in the
		// bulge and an effective size (as in Lacey+16).
		if(mass_ratio >= parameters.minor_merger_burst_ratio and mgas_ratio > parameters.gas_fraction_burst_ratio){

			mtotal_central = central->bulge_mass() + central->disk_gas.mass;
			rcentral = (central->bulge_size() * central->bulge_mass() + central->disk_gas.mass * central->disk_gas.rscale) / mtotal_central;
		}
		else{

			mtotal_central = central->bulge_mass();
			rcentral = central->bulge_size();
		}

	}

	double r = r_remnant(mtotal_central, mbar_satellite, rcentral, rsatellite);

	if((std::isnan(r) or r <= 0 or r > 3) and (mtotal_central > 0 or mbar_satellite > 0)){
		std::ostringstream os;
		os << central << " has a bulge size not well defined in galaxy mergers.";
		throw invalid_data(os.str());
	}

	/**Shrink the sizes depending on the gas fraction of the merger as in Hopkins et al. (2009) and above some mass ratio
	set by the user.**/
	if(parameters.fgas_dissipation > 0 and mass_ratio > parameters.merger_ratio_dissipation){

		double mstars = central->stellar_mass() + satellite->stellar_mass();
		double mgas = central->gas_mass() + satellite->gas_mass();
		double rnew = r;

		if(mgas > 0 and mstars > 0){
			double rgas_gal = mgas / mstars;
			double denom = (1.0 + rgas_gal/parameters.fgas_dissipation);
			if(denom > 3){
				denom = 3;
			}
			rnew  = r / denom;
		}
		else if (mstars == 0 and mgas > 0){
			//allow a maximum change of a factor of 10.
			rnew  = r / 3.0;
		}

		if(rnew <= 0){
			std::ostringstream os;
			os << central << " galaxy has rbulge <= 0" << rnew;
			throw exception(os.str());
		}

		r = rnew;
	}

	if(r <= constants::EPS6){
		std::ostringstream os;
		os << "Galaxy with extremely small size, rbulge_gas < 1-6, in galaxy mergers";
		//throw invalid_argument(os.str());
	}


	return r;

}


double GalaxyMergers::r_remnant(double mc, double ms, double rc, double rs){

	/**
	 * Input variables:
	 * mc: mass central.
	 * ms: mass satellite.
	 * rc: radius central.
	 * rs: radius satellite.
	 */

	double factor1 = 0;

	if(rc > 0 and mc >0){
		factor1  = std::pow(mc,2.0)/rc;
	}
	double factor2 = 0;

	if(rs > 0 and ms > 0){
		factor2 = std::pow(ms,2.0)/rs;
	}

	double factor3 = 0;

	if(rc > 0 or rs > 0){
		factor3 = parameters.f_orbit/parameters.cgal *  mc * ms / (rc + rs);
	}

	double f = (factor1 + factor2 + factor3);

	double r = 0;

	if(f > 0){
		r = std::pow((mc + ms),2.0)/ f;
	}

	return r;
}

void GalaxyMergers::transfer_baryon_mass(SubhaloPtr central, SubhaloPtr satellite){

	central->hot_halo_gas += satellite->hot_halo_gas;
	central->cold_halo_gas += satellite->cold_halo_gas;
	central->ejected_galaxy_gas += satellite->ejected_galaxy_gas;

	// Make baryon components of satellite subhalo = 0.
	satellite->hot_halo_gas.restore_baryon();
	satellite->cold_halo_gas.restore_baryon();
	satellite->ejected_galaxy_gas.restore_baryon();

}

void GalaxyMergers::transfer_bulge_gas(SubhaloPtr &subhalo, GalaxyPtr &galaxy, double z){

	galaxy->disk_gas += galaxy->bulge_gas;

	if(galaxy->disk_gas.rscale == 0){
		galaxy->disk_gas.rscale = galaxy->bulge_gas.rscale;
		galaxy->disk_gas.sAM    = galaxy->bulge_gas.sAM;

		if (std::isnan(galaxy->disk_gas.sAM) or std::isnan(galaxy->disk_gas.rscale)) {
			throw invalid_argument("rgas or sAM are NaN, cannot continue at galaxy mergers - transfer_bulge_gas");
		}
	}

	galaxy->bulge_gas.restore_baryon();

}

void GalaxyMergers::transfer_history_satellite_to_bulge(GalaxyPtr &central, GalaxyPtr &satellite, int snapshot){

	/**
	 * Function transfers the satellite stellar mass history to the bulge of the central galaxy.
	 */

	//Transfer history of stellar mass growth until the previous snapshot.
	for(int s=simparams.min_snapshot; s <= snapshot-1; s++) {

		auto it_sat = std::find_if(satellite->history.begin(), satellite->history.end(), [s](const HistoryItem &hitem) {
			return hitem.snapshot == s;
		});

		auto it_cen = std::find_if(central->history.begin(), central->history.end(), [s](const HistoryItem &hitem) {
			return hitem.snapshot == s;
		});

		/**There will be four cases:
			1) that both galaxies existed at snapshot s. In this case transfer history at this snapshot to central.
			2) that the satellite didn't exist but the central did. In this case do nothing.
			3) that the central didn't exist but the satellite did. In this create a new entry for the history of the central with the data of the satellite.
			4) none of the galaxies existed. In this case do nothing.
		**/
		if (it_sat == satellite->history.end() and it_cen == central->history.end()){ //neither satellite or central existed.
			//no-opt.
		}
		else if (it_sat == satellite->history.end() and it_cen != central->history.end()){ // satellite didn't exist but central did.
			//no-opt.
		}
		else if (it_sat != satellite->history.end() and it_cen == central->history.end()){ // central didn't exist but satellite did.
			auto hist_item = *it_sat;

			//transfer all data to the bulge, which is where all of this mass ends up being at.
			hist_item.sfr_bulge   += hist_item.sfr_disk;
			hist_item.sfr_z_bulge += hist_item.sfr_z_disk;

			//Make disk properties = 0.
			hist_item.sfr_disk   = 0;
			hist_item.sfr_z_disk = 0;

			central->history.push_back(hist_item);
		}
		else { // both galaxies exist at this snapshot
			auto hist_sat = *it_sat;
			auto &hist_cen = *it_cen;

			hist_cen.sfr_bulge   += hist_sat.sfr_bulge + hist_sat.sfr_disk;
			hist_cen.sfr_z_bulge += hist_sat.sfr_z_bulge + hist_sat.sfr_z_disk;

		}
	}

}

void GalaxyMergers::transfer_history_disk_to_bulge(GalaxyPtr &central, int snapshot){

	/**
	 * Function transfers the disk stellar mass history to bulge of the central galaxy.
	 */

	//Transfer history of stellar mass growth until the previous snapshot.
	for(int s=simparams.min_snapshot; s <= snapshot-1; s++) {

		auto it_cen = std::find_if(central->history.begin(), central->history.end(), [s](const HistoryItem &hitem) {
			return hitem.snapshot == s;
		});

		if (it_cen == central->history.end()){ //central didn't exist.
			//no-opt.
		}
		else { // both galaxies exist at this snapshot
			auto &hist_cen = *it_cen;

			//tranfer disk information to bulge.
			hist_cen.sfr_bulge   += hist_cen.sfr_disk;
			hist_cen.sfr_z_bulge += hist_cen.sfr_z_disk;

			//make disk properties = 0;
			hist_cen.sfr_disk = 0;
			hist_cen.sfr_z_disk = 0;
		}
	}


}


}  // namespace shark
