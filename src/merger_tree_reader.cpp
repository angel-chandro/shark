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

#include <array>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <tuple>

#include "dark_matter_halos.h"
#include "exceptions.h"
#include "halo.h"
#include "logging.h"
#include "merger_tree_reader.h"
#include "omp_utils.h"
#include "simulation.h"
#include "subhalo.h"
#include "timer.h"
#include "utils.h"
#include "hdf5/io/reader.h"


namespace shark {

  SURFSReader::SURFSReader(const std::string &prefix, DarkMatterHalosPtr dark_matter_halos, SimulationParameters simulation_params, unsigned int threads, const std::string &mask_prefix, const std::string &mask_name) :
    prefix(prefix), dark_matter_halos(std::move(dark_matter_halos)), simulation_params(std::move(simulation_params)), threads(threads), mask_prefix(mask_prefix),  mask_name(mask_name)
{
	if (prefix.empty()) {
		throw invalid_argument("Trees dir has no value");
	}

}

const std::string SURFSReader::get_filename(unsigned int batch)
{
	std::ostringstream os;
	os << prefix << "." << batch << ".hdf5";
	return os.str();
}

const std::string SURFSReader::get_mask_filename(unsigned int batch)
{
	std::ostringstream os;
	os << mask_prefix << "." << batch << ".hdf5";
	return os.str();
}

  
const std::vector<HaloPtr> SURFSReader::read_halos(std::vector<unsigned int> batches)
{

	// Check that batch numbers are within boundaries
	// (supposing that the file for batch 0 always exists)
	unsigned int nbatches;
	{
		auto batch0_fname = get_filename(0);
		LOG(debug) << "Opening " << batch0_fname << " for reading";
		hdf5::Reader batchfile_0(batch0_fname);
		nbatches = batchfile_0.read_attribute<unsigned int>("fileInfo/numberOfFiles");
	}

	for(auto batch: batches) {
		if (batch >= nbatches) {
			std::ostringstream os;
			os << "Batch is greater than numberOfFile specified in " << get_filename(0);
			os << ": " << batch << " > " << nbatches;
			throw invalid_argument(os.str());
		}
	}

	// Read halos for each batch, accumulate and return
	std::vector<HaloPtr> all_halos;
	for(auto batch: batches) {
		LOG(info) << "Reading file for batch " << batch;
		auto halos_batch = read_halos(batch);
		all_halos.reserve(all_halos.size() + halos_batch.size());
		all_halos.insert(all_halos.end(), halos_batch.begin(), halos_batch.end());
	}

	return all_halos;
}

const std::vector<SubhaloPtr> SURFSReader::read_subhalos(unsigned int batch)
{
	Timer t;
	const auto fname = get_filename(batch);
	hdf5::Reader batch_file(fname);

	std::vector<float> position;
	std::vector<float> velocity;
	std::vector<float> Mvir;
	std::vector<int> Npart;
	std::vector<float> Vcirc;
	std::vector<float> L;
	std::vector<float> Mgas;
	std::vector<int> snap;
	std::vector<Subhalo::id_t> nodeIndex;
	std::vector<Subhalo::id_t> descIndex;
	std::vector<Halo::id_t> hostIndex;
	std::vector<Halo::id_t> descHost;
	std::vector<int> IsMain;
	std::vector<int> IsCentre;
	std::vector<int> IsInterpolated;

	if (mask_prefix == "none") {
  
	  //Read position and velocities first.
	  position = batch_file.read_dataset_v_2<float>("haloTrees/position");
	  velocity = batch_file.read_dataset_v_2<float>("haloTrees/velocity");
	  
	  //Read mass, npart, circular velocity and angular momentum.
	  Mvir = batch_file.read_dataset_v<float>("haloTrees/nodeMass");
	  Npart = batch_file.read_dataset_v<int>("haloTrees/particleNumber");
	  Vcirc = batch_file.read_dataset_v<float>("haloTrees/maximumCircularVelocity");
	  L = batch_file.read_dataset_v_2<float>("haloTrees/angularMomentum");

	  if (simulation_params.hydrorun) {
	    Mgas = batch_file.read_dataset_v<float>("haloTrees/Mgas");
	  }

	  //Read indices and the snapshot number at which the subhalo lives.
	  snap = batch_file.read_dataset_v<int>("haloTrees/snapshotNumber");
	  nodeIndex = batch_file.read_dataset_v<Subhalo::id_t>("haloTrees/nodeIndex");
	  descIndex = batch_file.read_dataset_v<Subhalo::id_t>("haloTrees/descendantIndex");
	  hostIndex = batch_file.read_dataset_v<Halo::id_t>("haloTrees/hostIndex");
	  descHost = batch_file.read_dataset_v<Halo::id_t>("haloTrees/descendantHost");
	  
	  //Read properties that characterise the position of the subhalo inside the halo.descendantIndex
	  IsMain = batch_file.read_dataset_v<int>("haloTrees/isMainProgenitor");
	  IsCentre = batch_file.read_dataset_v<int>("haloTrees/isDHaloCentre");
	  IsInterpolated = batch_file.read_dataset_v<int>("haloTrees/isInterpolated");
	}
	  
	else {

	  const auto fname_mask = get_mask_filename(batch);
	  hdf5::Reader batch_mask(fname_mask);
	  
	  std::vector<float> positionBef = batch_file.read_dataset_v_2<float>("haloTrees/position");
	  std::vector<float> velocityBef = velocity = batch_file.read_dataset_v_2<float>("haloTrees/velocity");
	  std::vector<float> MvirBef = batch_file.read_dataset_v<float>("haloTrees/nodeMass");
	  std::vector<int> NpartBef = batch_file.read_dataset_v<int>("haloTrees/particleNumber");;
	  std::vector<float> VcircBef = batch_file.read_dataset_v<float>("haloTrees/maximumCircularVelocity");
	  std::vector<float> LBef = batch_file.read_dataset_v_2<float>("haloTrees/angularMomentum");
	  std::vector<float> MgasBef;
	  if (simulation_params.hydrorun) {
	    MgasBef = batch_file.read_dataset_v<float>("haloTrees/Mgas");
	  }
	  std::vector<int> snapBef = batch_file.read_dataset_v<int>("haloTrees/snapshotNumber");
	  std::vector<Subhalo::id_t> nodeIndexBef = batch_file.read_dataset_v<Subhalo::id_t>("haloTrees/nodeIndex");
	  std::vector<Subhalo::id_t> descIndexBef = batch_file.read_dataset_v<Subhalo::id_t>("haloTrees/descendantIndex");
	  std::vector<Halo::id_t> hostIndexBef = batch_file.read_dataset_v<Halo::id_t>("haloTrees/hostIndex");
	  std::vector<Halo::id_t> descHostBef = batch_file.read_dataset_v<Halo::id_t>("haloTrees/descendantHost");
	  std::vector<int> IsMainBef = batch_file.read_dataset_v<int>("haloTrees/isMainProgenitor");
	  std::vector<int> IsCentreBef = batch_file.read_dataset_v<int>("haloTrees/isDHaloCentre");
	  std::vector<int> IsInterpolatedBef = IsInterpolated = batch_file.read_dataset_v<int>("haloTrees/isInterpolated");

	  // opening mask file
	  std::vector<Subhalo::id_t> nodeIndexMask = batch_mask.read_dataset_v<Subhalo::id_t>("nodeIndex");
	  std::vector<int> mask_array = batch_mask.read_dataset_v<int>(mask_name);

	  // saving only selected IDs
	  std::set<Subhalo::id_t> nodes_to_ignore;
	  for (int i = 0; i < nodeIndexMask.size(); i++) {
	    if (mask_array[i]) {
	      nodes_to_ignore.insert(nodeIndexMask[i]);
	    }
	  }

	  // remove 0 values
	  nodes_to_ignore.erase(0);
	  
	  // then use nodes_to_ignore.count(id) when reading subhalos
	  for (int i = 0; i < nodeIndexBef.size() ; i++){
	    if (!nodes_to_ignore.count(nodeIndexBef[i])) {
		//Read position and velocities first.
		position.push_back(positionBef[i]);
		velocity.push_back(velocityBef[i]);
		//Read mass, npart, circular velocity and angular momentum.
		Mvir.push_back(MvirBef[i]);
		Npart.push_back(NpartBef[i]);
		Vcirc.push_back(VcircBef[i]);
		L.push_back(LBef[i]);
		if (simulation_params.hydrorun) {
		  Mgas.push_back(MgasBef[i]);
		}
		//Read indices and the snapshot number at which the subhalo lives.
		snap.push_back(snapBef[i]);
		nodeIndex.push_back(nodeIndexBef[i]);
 		descIndex.push_back(descIndexBef[i]);
 		hostIndex.push_back(hostIndexBef[i]);
		descHost.push_back(descHostBef[i]);
		//Read properties that characterise the position of the subhalo inside the halo.descendantIndex
		IsMain.push_back(IsMainBef[i]);
		IsCentre.push_back(IsCentreBef[i]);
		IsInterpolated.push_back(IsInterpolatedBef[i]);

	    }
	  }

	}
	    
	auto n_subhalos = Mvir.size();
	LOG(info) << "Read raw data of " << n_subhalos << " subhalos from " << fname << " in " << t;
	if (n_subhalos == 0) {
		return {};
	}

	std::ostringstream os;
	os << "File " << fname << " has " << n_subhalos << " subhalos. ";
	os << "After reading we should be using ~" << memory_amount(n_subhalos * (sizeof(Subhalo) + sizeof(SubhaloPtr))) << " of memory";
	LOG(info) << os.str();

	t = Timer();
	std::vector<std::vector<SubhaloPtr>> t_subhalos(threads);
	for (auto &subhalos: t_subhalos) {
		subhalos.reserve(n_subhalos / threads);
	}

	omp_static_for(0, n_subhalos, threads, [&](std::size_t i, unsigned int thread_idx) {

		if (snap[i] < simulation_params.min_snapshot) {
			return;
		}

		//Check that this subhalo has a DM mass > 0 in the case of hydrodynamical simulation input. The latter can happen at the resolution limit.
		//If gas mass is larger than total virial mass, then skip this subhalo.
		if(simulation_params.hydrorun){
			if(Mvir[i]-Mgas[i] < 0){
			  return;
			}
		}
			
		auto subhalo = std::make_shared<Subhalo>(nodeIndex[i], snap[i]);

		// Subhalo and Halo index, snapshot
		subhalo->haloID = hostIndex[i];

		// Descendant information. -1 means that the Subhalo has no descendant
		auto descendant_id = descIndex[i];
		if (descendant_id == -1) {
			subhalo->has_descendant = false;
		}
		else {
			subhalo->has_descendant = true;
			subhalo->descendant_id = descendant_id;
			subhalo->descendant_halo_id = descHost[i];
		}

		//Assign main progenitor flags.
		if(IsMain[i] == 1){
			subhalo->main_progenitor = true;
		}

		//Assign interpolated subhalo flags.
		if(IsInterpolated[i] == 1){
			subhalo->IsInterpolated = true;
		}

		//Make all subhalos satellite, because once we construct the merger tree we will find the main branch.
		subhalo->subhalo_type = Subhalo::SATELLITE;

		//Assign mass.
		subhalo->Mvir = Mvir[i];

		//Assign gas mass if the simulation is a hydrodynamical simulation.
		if(simulation_params.hydrorun){
		        subhalo->Mgas = Mgas[i];
		}

		//Assign npart
		subhalo->Npart = Npart[i];

		//Assign position
		subhalo->position.x = position[3 * i];
		subhalo->position.y = position[3 * i + 1];
		subhalo->position.z = position[3 * i + 2];

		//Assign velocity
		subhalo->velocity.x = velocity[3 * i];
		subhalo->velocity.y = velocity[3 * i + 1];
		subhalo->velocity.z = velocity[3 * i + 2];

		//Assign specific angular momentum
		subhalo->L.x = L[3 * i];
		subhalo->L.y = L[3 * i + 1];
		subhalo->L.z = L[3 * i + 2];

		subhalo->Vcirc = Vcirc[i];

		auto z = simulation_params.redshifts[subhalo->snapshot];
		subhalo->concentration = dark_matter_halos->nfw_concentration(subhalo->Mvir, z);

		if (subhalo->concentration < 1) {
			throw invalid_argument("concentration is <1, cannot continue. Please check input catalogue");
		}

		double npart = Mvir[i]/simulation_params.particle_mass;

		subhalo->lambda = dark_matter_halos->halo_lambda(*subhalo, Mvir[i], z, npart);

		// Calculate virial velocity from the virial mass and redshift.
		subhalo->Vvir = dark_matter_halos->halo_virial_velocity(subhalo->Mvir, z);

		// Done, save it now
		t_subhalos[thread_idx].emplace_back(std::move(subhalo));
	});

	std::vector<SubhaloPtr> subhalos;
	if (threads == 0) {
		subhalos = std::move(t_subhalos[0]);
	}
	else {
		subhalos.reserve(n_subhalos);
		for (auto i = 0u; i != threads; i++) {
			subhalos.insert(subhalos.end(), t_subhalos[i].begin(), t_subhalos[i].end());
		}
	}

	LOG(info) << "Created " << subhalos.size() << " Subhalos from " << fname << " in " << t;
	return subhalos;
}

const std::vector<HaloPtr> SURFSReader::read_halos(unsigned int batch)
{

	std::vector<SubhaloPtr> subhalos = read_subhalos(batch);

	// Sort subhalos by host index (which intrinsically sorts them by snapshot
	// since host indices numbers are prefixed with the snapshot number)
	std::sort(subhalos.begin(), subhalos.end(), [](const SubhaloPtr &lhs, const SubhaloPtr &rhs) {
		return lhs->haloID < rhs->haloID;
	});
	LOG(info) << "Sorted subhalos by haloID, creating Halos now";

	// Create and assign Halos
	HaloPtr halo;
	std::vector<HaloPtr> halos;
	Halo::id_t last_halo_id = -1;
	Timer t;
	for(auto &subhalo: subhalos) {

		auto halo_id = subhalo->haloID;
		if (halo_id != last_halo_id) {
			if (halo) {
				halos.emplace_back(std::move(halo));
			}
			last_halo_id = halo_id;
			halo = std::make_shared<Halo>(halo_id, subhalo->snapshot);
		}

		if (LOG_ENABLED(trace)) {
			LOG(trace) << "Adding " << subhalo << " to " << halo;
		}
		subhalo->host_halo = halo;
		halo->add_subhalo(std::move(subhalo));
	}
	subhalos.clear();

	std::ostringstream os;
	os << "Created " << halos.size() << " Halos from these Subhalos in " << t << ". ";
	os << "This should take another ~" << memory_amount(halos.size() * (sizeof(Halo) + sizeof(HaloPtr))) << " of memory";
	LOG(info) << os.str();

	// Calculate halos' vvir and concentration
	t = Timer();
	omp_dynamic_for(halos, threads, 10000, [&](const HaloPtr &halo, unsigned int thread_idx) {
		auto z = simulation_params.redshifts[halo->snapshot];
		halo->Vvir = dark_matter_halos->halo_virial_velocity(halo->Mvir, z);
		halo->concentration = dark_matter_halos->nfw_concentration(halo->Mvir,z);
	});
	LOG(info) << "Calculated Vvir and concentration for new Halos in " << t;

	return halos;
}

}  // namespace shark
