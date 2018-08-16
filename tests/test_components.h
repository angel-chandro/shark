//
// Components unit tests
//
// ICRAR - International Centre for Radio Astronomy Research
// (c) UWA - The University of Western Australia, 2018
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

#include <cxxtest/TestSuite.h>

#include <functional>

#include "components.h"
#include "exceptions.h"

using namespace shark;

class TestBaryons : public CxxTest::TestSuite {

public:

	void _init_baryons(Baryon &b1, Baryon &b2) {
		b1.mass = 1.;
		b1.mass_metals = 2.;
		b2.mass = 3.;
		b2.mass_metals = 4.;
	}

	void _assert_addition(const Baryon &b) {
		TS_ASSERT_DELTA(b.mass, 4., 1e-8);
		TS_ASSERT_DELTA(b.mass_metals, 6., 1e-8);
	}

	void test_baryons_compound_addition() {
		Baryon b1, b2;
		_init_baryons(b1, b2);
		b2 += b1;
		_assert_addition(b2);
	}

	void test_baryons_addition() {
		Baryon b1, b2;
		_init_baryons(b1, b2);
		Baryon b3 = b2 + b1;
		_assert_addition(b3);
	}

};

class TestSubhalos : public CxxTest::TestSuite
{
private:

	SubhaloPtr make_subhalo(const std::string &types, Subhalo::subhalo_type_t subhalo_type)
	{
		SubhaloPtr subhalo = std::make_shared<Subhalo>(0, 0);
		subhalo->subhalo_type = subhalo_type;
		Galaxy::id_t id = 0;
		for(auto t: types) {
			GalaxyPtr g = std::make_shared<Galaxy>(id++);
			if (t == 'C') {
				g->galaxy_type = Galaxy::CENTRAL;
			}
			else if (t == '1') {
				g->galaxy_type = Galaxy::TYPE1;
			}
			else if (t == '2') {
				g->galaxy_type = Galaxy::TYPE2;
			}
			subhalo->galaxies.emplace_back(std::move(g));
		}
		return subhalo;
	}

	template <typename SpecificCheck>
	void _test_valid_galaxy_composition(const std::string &galaxy_types, Subhalo::subhalo_type_t subhalo_type, SpecificCheck specific_check, bool valid)
	{
		auto subhalo = make_subhalo(galaxy_types, subhalo_type);
		if (!valid) {
			TSM_ASSERT_THROWS(galaxy_types, subhalo->check_subhalo_galaxy_composition(), invalid_data);
			TS_ASSERT_THROWS(specific_check(subhalo), invalid_data);
		}
		else {
			subhalo->check_subhalo_galaxy_composition();
			specific_check(subhalo);
		}
	}

	void _test_valid_central_galaxy_composition(const std::string &types, bool valid)
	{
		_test_valid_galaxy_composition(types, Subhalo::CENTRAL, std::mem_fn(&Subhalo::check_central_subhalo_galaxy_composition), valid);
	}

	void _test_valid_satellite_galaxy_composition(const std::string &types, bool valid)
	{
		_test_valid_galaxy_composition(types, Subhalo::SATELLITE, std::mem_fn(&Subhalo::check_satellite_subhalo_galaxy_composition), valid);
	}

public:

	void test_valid_central_galaxy_composition()
	{
		_test_valid_central_galaxy_composition("C", true);
		_test_valid_central_galaxy_composition("1", false);
		_test_valid_central_galaxy_composition("2", false);

		_test_valid_central_galaxy_composition("CC", false);
		_test_valid_central_galaxy_composition("C1", false);
		_test_valid_central_galaxy_composition("C2", true);
		_test_valid_central_galaxy_composition("11", false);
		_test_valid_central_galaxy_composition("12", false);
		_test_valid_central_galaxy_composition("22", false);

		_test_valid_central_galaxy_composition("C22", true);
		_test_valid_central_galaxy_composition("C22222", true);
		_test_valid_central_galaxy_composition("C222221", false);

		_test_valid_central_galaxy_composition("122", false);
		_test_valid_central_galaxy_composition("122222", false);
		_test_valid_central_galaxy_composition("122222C", false);
	}

	void test_valid_satellite_galaxy_composition()
	{
		_test_valid_satellite_galaxy_composition("C", false);
		_test_valid_satellite_galaxy_composition("1", true);
		_test_valid_satellite_galaxy_composition("2", true);

		_test_valid_satellite_galaxy_composition("CC", false);
		_test_valid_satellite_galaxy_composition("C1", false);
		_test_valid_satellite_galaxy_composition("C2", false);
		_test_valid_satellite_galaxy_composition("11", false);
		_test_valid_satellite_galaxy_composition("12", true);
		_test_valid_satellite_galaxy_composition("22", true);

		_test_valid_satellite_galaxy_composition("C22", false);
		_test_valid_satellite_galaxy_composition("C22222", false);
		_test_valid_satellite_galaxy_composition("C222221", false);

		_test_valid_satellite_galaxy_composition("122", true);
		_test_valid_satellite_galaxy_composition("122222", true);
		_test_valid_satellite_galaxy_composition("122222C", false);
	}

};

class TestHalo : public CxxTest::TestSuite
{

public:

	template <typename Iterator>
	void _assert_different(const Iterator &it1, const Iterator &it2)
	{
		TS_ASSERT(it1 != it2);
		TS_ASSERT(!(it1 == it2));
	}

	template <typename Iterator>
	void _assert_equals(const Iterator &it1, const Iterator &it2)
	{
		TS_ASSERT(it1 == it2);
		TS_ASSERT(!(it1 != it2));
	}

	void test_subhalos_view_empty_halo()
	{
		// Empty Halo
		Halo h(1, 1);
		auto all = Halo::subhalos_view<Halo>(h);
		_assert_equals(all.begin(), all.end());
	}

	void test_subhalos_view_only_central_subhalo_halo()
	{
		// A halo only with a central subhalo
		Halo h(1, 1);
		h.central_subhalo = std::make_shared<Subhalo>(0, 1);
		auto all = Halo::subhalos_view<Halo>(h);
		auto it = all.begin();
		_assert_different(it, all.end());
		_assert_equals(++it, all.end());
	}

	void test_subhalos_view_only_satellite_subhalos_halo()
	{
		// Only satellites
		int n_satellites = 10;
		Halo h(1, 1);
		for (int i = 0; i != n_satellites; i++) {
			auto subhalo = std::make_shared<Subhalo>(i, 1);
			subhalo->subhalo_type = Subhalo::SATELLITE;
			h.add_subhalo(std::move(subhalo));
		}

		auto all = Halo::subhalos_view<Halo>(h);
		auto it = all.begin();
		for (int i = 0; i != n_satellites - 1; i++) {
			_assert_different(it, all.end());
			++it;
		}
		_assert_equals(++it, all.end());
	}

	void test_subhalos_view_normal_halo()
	{
		// Central subhalo and satellites
		int n_satellites = 10;
		Halo h(1, 1);
		h.central_subhalo = std::make_shared<Subhalo>(0, 1);
		for (int i = 0; i != n_satellites; i++) {
			auto subhalo = std::make_shared<Subhalo>(i, 1);
			subhalo->subhalo_type = Subhalo::SATELLITE;
			h.add_subhalo(std::move(subhalo));
		}

		auto all = Halo::subhalos_view<Halo>(h);
		auto it = all.begin();
		for (int i = 0; i != n_satellites; i++) {
			_assert_different(it, all.end());
			++it;
		}
		_assert_equals(++it, all.end());
	}

};