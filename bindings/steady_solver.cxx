#include "steady_solver.hxx"

#include <HygroThermFEM/HygroThermFEM.hxx>
#include "DomainErrorEstimate.hxx"
#include "model_inputs.hxx"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace pyhygrotherm
{
    namespace
    {
        using detail::Edge;
        using detail::edge;
        using detail::finite;
    }   // namespace

    SteadySolution solve_steady(const std::vector<NodeInput> & nodes,
                                const std::vector<QuadInput> & quads,
                                const std::vector<MaterialInput> & materials,
                                const std::vector<ConvectionInput> & convection,
                                const std::vector<RadiationInput> & radiations,
                                const std::vector<FluxInput> & fluxes,
                                const std::vector<TemperatureInput> & temperatures,
                                const std::array<double, 3> & gravity,
                                bool estimate_error,
                                const std::vector<CavitySurfaceEmissivityInput> &
                                  cavity_surface_emissivities)
    {
        if(nodes.size() < 4 || quads.empty())
        {
            throw std::invalid_argument("at least four nodes and one Q4 element are required");
        }
        if(materials.size() != quads.size())
        {
            throw std::invalid_argument("one material is required for each Q4 element");
        }

        const auto fem_id = detail::femIndexMap(nodes);

        const auto lookup = [&](std::size_t id) {
            const auto found = fem_id.find(id);
            if(found == fem_id.end())
            {
                throw std::invalid_argument("element or boundary references an unknown node id");
            }
            return found->second;
        };

        HygroThermFEM::MultiDomain model({.performThermal = true, .performMoisture = false});
        if(!finite({gravity[0], gravity[1], gravity[2]})
           || std::abs(gravity[0] * gravity[0] + gravity[1] * gravity[1]
                       + gravity[2] * gravity[2] - 1.0) > 1e-12)
        {
            throw std::invalid_argument("gravity must be a finite unit vector");
        }
        model.setGravityVector({gravity[0], gravity[1], gravity[2]});
        model.setPhysicsOptions({.excludeHeatOfEvaporation = true,
                                 .excludeCapillaryConduction = true,
                                 .excludeVaporDiffusionConduction = true});
        constexpr double tolerance = 1e-10;
        model.setSolverSettings({.relaxationParameter = 1.0,
                                 .errorTolerance = tolerance,
                                 .maxNumberOfIterations = 32,
                                 .maxDivisions = 8,
                                 .numberOfSubtimesteps = 8});

        std::map<Edge, std::size_t> mesh_edges;
        std::map<MaterialInput, std::string> material_names;
        for(std::size_t index = 0; index < quads.size(); ++index)
        {
            const auto & quad = quads[index];
            const auto & spec = materials[index];
            const auto & [conductivity, emissivity1, emissivity2, standard, gas,
                          key, local_emissivities, cavity_height, cavity_pressure] = spec;
            if(!std::isfinite(emissivity1) || emissivity1 < 0 || emissivity1 > 1
               || !std::isfinite(emissivity2) || emissivity2 < 0 || emissivity2 > 1)
            {
                throw std::invalid_argument("material emissivities must be in [0, 1]");
            }
            std::set<std::size_t> distinct(quad.begin(), quad.end());
            if(distinct.size() != 4)
            {
                throw std::invalid_argument("each Q4 element must have four distinct node ids");
            }
            for(const auto id : quad)
            {
                (void)lookup(id);
            }
            for(std::size_t index = 0; index < quad.size(); ++index)
            {
                ++mesh_edges[edge(quad[index], quad[(index + 1) % quad.size()])];
            }
            if(!material_names.contains(spec))
            {
                const auto name = "material " + std::to_string(material_names.size() + 1);
                if(standard.empty())
                {
                    if(!std::isfinite(conductivity) || conductivity <= 0.0)
                    {
                        throw std::invalid_argument("conductivities must be finite and positive");
                    }
                    auto & material = model.materials().createSolidMaterial(name);
                    material.setThermalConductivity(conductivity);
                    material.setEmissivity(emissivity1);
                }
                else
                {
                    HygroThermFEM::CavityStandard cavity_standard;
                    if(standard == "CEN")
                    {
                        cavity_standard = HygroThermFEM::CavityStandard::CEN;
                    }
                    else if(standard == "ISO15099")
                    {
                        cavity_standard = HygroThermFEM::CavityStandard::ISO15099;
                    }
                    else if(standard == "ISO15099Ventilated")
                    {
                        cavity_standard = HygroThermFEM::CavityStandard::ISO15099Ventilated;
                    }
                    else
                    {
                        throw std::invalid_argument("unsupported cavity standard " + standard);
                    }
                    Gases::GasDef gas_type;
                    if(gas == "Air") gas_type = Gases::GasDef::Air;
                    else if(gas == "Argon") gas_type = Gases::GasDef::Argon;
                    else if(gas == "Krypton") gas_type = Gases::GasDef::Krypton;
                    else if(gas == "Xenon") gas_type = Gases::GasDef::Xenon;
                    else throw std::invalid_argument("unsupported cavity gas " + gas);
                    Gases::CGas fill;
                    fill.addGasItem(1.0, gas_type);
                    model.materials().createGas(name, cavity_standard, fill);
                    model.materials().gas(name).setCavityConditions(cavity_height,
                                                                   cavity_pressure);
                    if(!local_emissivities)
                    {
                        model.materials().gas(name).setSideEmissivities(emissivity1,
                                                                      emissivity2);
                    }
                }
                material_names.emplace(spec, name);
            }
        }

        for(std::size_t index = 0; index < nodes.size(); ++index)
        {
            const auto [id, x, y] = nodes[index];
            model.nodes().createNode({.index = index + 1, .x = x, .y = y});
        }

        for(std::size_t index = 0; index < quads.size(); ++index)
        {
            const auto & quad = quads[index];
            model.createElement({.node1 = lookup(quad[0]),
                                 .node2 = lookup(quad[1]),
                                 .node3 = lookup(quad[2]),
                                 .node4 = lookup(quad[3]),
                                 .material = material_names.at(materials[index])});
        }

        std::set<Edge> cavity_surface_edges;
        for(const auto & [node1, node2, emissivity] : cavity_surface_emissivities)
        {
            const auto boundary = edge(node1, node2);
            if(node1 == node2 || !mesh_edges.contains(boundary))
            {
                throw std::invalid_argument(
                  "cavity surface emissivity must match a Q4 mesh edge");
            }
            if(!std::isfinite(emissivity) || emissivity < 0.0 || emissivity > 1.0)
            {
                throw std::invalid_argument(
                  "cavity surface emissivity must be in [0, 1]");
            }
            if(!cavity_surface_edges.insert(edge(lookup(node1), lookup(node2))).second)
            {
                throw std::invalid_argument(
                  "cavity surface emissivity edges must be unique");
            }
            model.setCavitySurfaceEmissivity(lookup(node1), lookup(node2), emissivity);
        }

        bool anchored = false;
        std::set<Edge> convection_edges;
        for(const auto & [node1, node2, h, air_temperature1, air_temperature2] : convection)
        {
            const auto boundary = edge(node1, node2);
            if(node1 == node2 || !mesh_edges.contains(boundary) || mesh_edges.at(boundary) != 1)
            {
                throw std::invalid_argument("convection boundary must match a Q4 exterior edge");
            }
            if(!convection_edges.insert(boundary).second)
            {
                throw std::invalid_argument("convection boundary edges must be unique");
            }
            if(!std::isfinite(h) || h < 0.0 || !std::isfinite(air_temperature1)
               || !std::isfinite(air_temperature2))
            {
                throw std::invalid_argument("convection values must be finite and h non-negative");
            }
            if(air_temperature1 != air_temperature2)
            {
                throw std::invalid_argument("ambient temperature must be uniform on each convection edge");
            }
            if(h > 0.0)
            {
                model.thermal().createBC_FixedHc(lookup(node1),
                                                 lookup(node2),
                                                 {air_temperature1, h},
                                                 false);
                anchored = true;
            }
        }

        std::set<Edge> radiation_edges;
        for(const auto & [node1, node2, emissivity, radiation_temperature] : radiations)
        {
            const auto boundary = edge(node1, node2);
            if(node1 == node2 || !mesh_edges.contains(boundary) || mesh_edges.at(boundary) != 1)
            {
                throw std::invalid_argument("radiation boundary must match a Q4 exterior edge");
            }
            if(!radiation_edges.insert(boundary).second)
            {
                throw std::invalid_argument("radiation boundary edges must be unique");
            }
            if(!std::isfinite(emissivity) || emissivity < 0.0 || emissivity > 1.0
               || !std::isfinite(radiation_temperature))
            {
                throw std::invalid_argument("radiation emissivity must be in [0, 1] and temperature finite");
            }
            if(emissivity > 0.0)
            {
                model.thermal().createBC_BlackBodyRadiation(lookup(node1),
                                                            lookup(node2),
                                                            emissivity,
                                                            radiation_temperature);
                anchored = true;
            }
        }

        std::set<Edge> flux_edges;
        for(const auto & [node1, node2, flux] : fluxes)
        {
            const auto boundary = edge(node1, node2);
            if(node1 == node2 || !mesh_edges.contains(boundary) || mesh_edges.at(boundary) != 1)
            {
                throw std::invalid_argument("heat-flux boundary must match a Q4 exterior edge");
            }
            if(!flux_edges.insert(boundary).second)
            {
                throw std::invalid_argument("heat-flux boundary edges must be unique");
            }
            if(!std::isfinite(flux))
            {
                throw std::invalid_argument("heat flux must be finite");
            }
            model.thermal().createBC_FixedFlux(lookup(node1), lookup(node2), flux);
        }

        std::set<Edge> temperature_edges;
        for(const auto & [node1, node2, temperature] : temperatures)
        {
            const auto boundary = edge(node1, node2);
            if(node1 == node2 || !mesh_edges.contains(boundary) || mesh_edges.at(boundary) != 1)
            {
                throw std::invalid_argument("temperature boundary must match a Q4 exterior edge");
            }
            if(!temperature_edges.insert(boundary).second
               || convection_edges.contains(boundary)
               || radiation_edges.contains(boundary))
            {
                throw std::invalid_argument("temperature boundary edges must be unique and exclusive");
            }
            if(!std::isfinite(temperature))
            {
                throw std::invalid_argument("temperature boundary value must be finite");
            }
            model.thermal().createBC_FixedTemperature(lookup(node1), lookup(node2), temperature);
            anchored = true;
        }

        if(!anchored)
        {
            throw std::invalid_argument("steady-state model requires convection, radiation or fixed temperature to anchor temperature");
        }

        const auto solution = model.steadyState();
        if(!solution.converged || !std::isfinite(solution.temperatureError)
           || solution.temperatureError > tolerance)
        {
            throw std::runtime_error("steady-state solve did not converge");
        }
        if(solution.temperature.size() != nodes.size() || solution.heatFlux.size() != nodes.size()
           || !finite(solution.temperature))
        {
            throw std::runtime_error("steady-state solve returned incomplete or non-finite results");
        }

        SteadySolution result;
        result.temperature = solution.temperature;
        if(estimate_error)
        {
            const auto estimate = HygroThermFEM::errorest::estimateError(model.thermal(), 10.0);
            if(estimate.result.elementRelativeError.size() != quads.size()
               || !std::isfinite(estimate.result.globalErrorPercent))
            {
                throw std::runtime_error("energy error estimation failed");
            }
            result.energy_error_percent = estimate.result.globalErrorPercent;
        }
        result.heat_flux.reserve(solution.heatFlux.size());
        for(const auto & flux : solution.heatFlux)
        {
            if(!std::isfinite(flux.x) || !std::isfinite(flux.y))
            {
                throw std::runtime_error("steady-state solve returned non-finite heat flux");
            }
            result.heat_flux.push_back({flux.x, flux.y});
        }
        return result;
    }
}   // namespace pyhygrotherm
