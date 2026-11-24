#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

namespace pyhygrotherm
{
    using NodeInput = std::tuple<std::size_t, double, double>;
    using QuadInput = std::array<std::size_t, 4>;
    // conductivity, side emissivities, standard, gas, region key,
    // use local surfaces, cavity height (m), pressure (Pa).
    using MaterialInput = std::tuple<double, double, double, std::string,
                                     std::string, std::string, bool, double, double>;
    using ConvectionInput = std::tuple<std::size_t, std::size_t, double, double, double>;
    using RadiationInput = std::tuple<std::size_t, std::size_t, double, double>;
    using FluxInput = std::tuple<std::size_t, std::size_t, double>;
    using TemperatureInput = std::tuple<std::size_t, std::size_t, double>;
    using CavitySurfaceEmissivityInput = std::tuple<std::size_t, std::size_t, double>;

    struct SteadySolution
    {
        std::vector<double> temperature;
        std::vector<std::array<double, 2>> heat_flux;
        double energy_error_percent = -1.0;
    };

    SteadySolution solve_steady(const std::vector<NodeInput> & nodes,
                                const std::vector<QuadInput> & quads,
                                const std::vector<MaterialInput> & materials,
                                const std::vector<ConvectionInput> & convection,
                                const std::vector<RadiationInput> & radiations,
                                const std::vector<FluxInput> & fluxes,
                                const std::vector<TemperatureInput> & temperatures,
                                const std::array<double, 3> & gravity = {0.0, -1.0, 0.0},
                                bool estimate_error = false,
                                const std::vector<CavitySurfaceEmissivityInput> &
                                  cavity_surface_emissivities = {});
}   // namespace pyhygrotherm
