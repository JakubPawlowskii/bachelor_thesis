#ifndef XXZ_REAL_H
#define XXZ_REAL_H
#pragma once

#include "hamiltonian.hpp"

class xxz_real : public hamiltonian<arma::mat>
{
	private:
		double t;
		double delta;
		double alpha;
	public:

		// xxz_real(uint64_t sites, double t, double delta);
		xxz_real(uint64_t sites, double t, double delta, double alpha);
		// xxz_real(uint64_t sites, uint64_t particles, double t, double delta);
		xxz_real(uint64_t sites, uint64_t particles, double t, double delta, double alpha);
		virtual ~xxz_real() = default;
		double get_alpha() const;
		double get_t() const;
		double get_delta() const;

		void regenerate();

};
#endif