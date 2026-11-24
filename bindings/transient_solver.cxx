#include "transient_solver.hxx"

#include "model_inputs.hxx"

#include <HygroThermFEM/HygroThermFEM.hxx>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace pyhygrotherm
{
    namespace
    {
        using detail::Edge;
        using detail::edge;

        // A timestep series on a boundary edge is either constant (one
        // entry, held for the whole run via the kernel's scalar BC
        // overloads) or periodic. Varying series attach through the kernel's
        // periodic BC overloads, which index every coefficient field by its
        // own period (value at step t is series[t % period]). Memory stays
        // proportional to the sum of the declared field periods -- combining
        // fields into a single lcm-period table would explode on coprime
        // periods and defeat the bounded-memory contract.
        template<typename Series>
        void check_series(const Series & series, const char * what)
        {
            if(series.empty())
            {
                throw std::invalid_argument(
                  std::string(what) + " boundary requires a timestep series");
            }
            for(const auto & entry : series)
            {
                if(!std::isfinite(entry))
                {
                    throw std::invalid_argument(
                      std::string(what) + " series values must be finite");
                }
            }
        }

        // Periodic kernel series view over a compact field series.
        HygroThermFEM::PeriodicSeries
          periodic(const std::vector<double> & series)
        {
            return HygroThermFEM::PeriodicSeries{series};
        }
    }   // namespace

    TransientSolution solve_transient(
      const std::vector<NodeInput> & nodes,
      const std::vector<QuadInput> & quads,
      const std::vector<MaterialInput> & materials,
      const std::vector<double> & densities,
      const std::vector<double> & heat_capacities,
      const std::vector<TimestepConvectionInput> & convection,
      const std::vector<TimestepTemperatureInput> & temperatures,
      const std::vector<TimestepFluxInput> & fluxes,
      const std::vector<TimestepRadiationInput> & radiations,
      const std::vector<TimestepLinearizedRadiationInput> & linearized_radiations,
      const double initial_temperature,
      const double initial_humidity,
      const double time_step,
      const std::size_t num_steps,
      const double convergence_tolerance,
      const double relaxation,
      const std::array<double, 3> & gravity,
      const std::vector<CavitySurfaceEmissivityInput> &
        cavity_surface_emissivities,
      const std::function<void(const TransientFrame &)> & on_step)
    {
        if(nodes.size() < 4 || quads.empty())
        {
            throw std::invalid_argument(
              "at least four nodes and one Q4 element are required");
        }
        if(materials.size() != quads.size() || densities.size() != quads.size()
           || heat_capacities.size() != quads.size())
        {
            throw std::invalid_argument(
              "one material, density and heat capacity is required for each "
              "Q4 element");
        }
        if(!std::isfinite(initial_temperature)
           || !std::isfinite(initial_humidity) || initial_humidity < 0.0
           || initial_humidity > 1.0)
        {
            throw std::invalid_argument(
              "initial temperature must be finite and humidity in [0, 1]");
        }
        if(!std::isfinite(time_step) || time_step <= 0.0 || num_steps == 0)
        {
            throw std::invalid_argument(
              "time step must be positive and at least one step is required");
        }
        if(!std::isfinite(convergence_tolerance) || convergence_tolerance <= 0.0
           || !std::isfinite(relaxation) || relaxation <= 0.0
           || relaxation > 1.0)
        {
            throw std::invalid_argument(
              "convergence tolerance must be positive and relaxation in (0, 1]");
        }

        const auto fem_id = detail::femIndexMap(nodes);
        const auto lookup = [&](std::size_t id) {
            const auto found = fem_id.find(id);
            if(found == fem_id.end())
            {
                throw std::invalid_argument(
                  "element or boundary references an unknown node id");
            }
            return found->second;
        };

        HygroThermFEM::MultiDomain model(
          {.performThermal = true, .performMoisture = false});
        if(!detail::finite({gravity[0], gravity[1], gravity[2]})
           || std::abs(gravity[0] * gravity[0] + gravity[1] * gravity[1]
                       + gravity[2] * gravity[2] - 1.0) > 1e-12)
        {
            throw std::invalid_argument("gravity must be a finite unit vector");
        }
        model.setGravityVector({gravity[0], gravity[1], gravity[2]});
        model.setPhysicsOptions({.excludeWaterLiquidTransportation = true,
                                  .excludeHeatOfEvaporation = true,
                                  .excludeCapillaryConduction = true,
                                  .excludeVaporDiffusionConduction = true,
                                  .excludeLatentHeatOfFusion = true});
        model.setSolverSettings({.relaxationParameter = relaxation,
                                  .errorTolerance = convergence_tolerance,
                                  .maxNumberOfIterations = 32,
                                  .maxDivisions = 8,
                                  .numberOfSubtimesteps = 8});

        std::map<Edge, std::size_t> mesh_edges;
        using TransientMaterialKey =
          std::tuple<MaterialInput, double, double>;
        std::map<TransientMaterialKey, std::string> material_names;
        for(std::size_t index = 0; index < quads.size(); ++index)
        {
            const auto & quad = quads[index];
            const auto & spec = materials[index];
            const auto density = densities[index];
            const auto heat_capacity = heat_capacities[index];
            const auto & [conductivity, emissivity1, emissivity2, standard,
                          gas, key, local_emissivities, cavity_height,
                          cavity_pressure] = spec;
            if(!std::isfinite(emissivity1) || emissivity1 < 0 || emissivity1 > 1
               || !std::isfinite(emissivity2) || emissivity2 < 0
               || emissivity2 > 1)
            {
                throw std::invalid_argument(
                  "material emissivities must be in [0, 1]");
            }
            std::set<std::size_t> distinct(quad.begin(), quad.end());
            if(distinct.size() != 4)
            {
                throw std::invalid_argument(
                  "each Q4 element must have four distinct node ids");
            }
            for(const auto id : quad)
            {
                (void)lookup(id);
            }
            for(std::size_t corner = 0; corner < quad.size(); ++corner)
            {
                ++mesh_edges[edge(quad[corner], quad[(corner + 1) % quad.size()])];
            }
            const TransientMaterialKey material_key{spec, density,
                                                    heat_capacity};
            if(!material_names.contains(material_key))
            {
                const auto name =
                  "material " + std::to_string(material_names.size() + 1);
                if(standard.empty())
                {
                    if(!std::isfinite(conductivity) || conductivity <= 0.0)
                    {
                        throw std::invalid_argument(
                          "conductivities must be finite and positive");
                    }
                    if(!std::isfinite(density) || density <= 0.0)
                    {
                        throw std::invalid_argument(
                          "solid densities must be finite and positive");
                    }
                    if(!std::isfinite(heat_capacity) || heat_capacity <= 0.0)
                    {
                        throw std::invalid_argument(
                          "solid heat capacities must be finite and positive");
                    }
                    auto & material =
                      model.materials().createSolidMaterial(name);
                    material.setThermalConductivity(conductivity);
                    material.setDensity(density);
                    material.setHeatCapacity(heat_capacity);
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
                        cavity_standard =
                          HygroThermFEM::CavityStandard::ISO15099Ventilated;
                    }
                    else
                    {
                        throw std::invalid_argument(
                          "unsupported cavity standard " + standard);
                    }
                    Gases::GasDef gas_type;
                    if(gas == "Air") gas_type = Gases::GasDef::Air;
                    else if(gas == "Argon") gas_type = Gases::GasDef::Argon;
                    else if(gas == "Krypton") gas_type = Gases::GasDef::Krypton;
                    else if(gas == "Xenon") gas_type = Gases::GasDef::Xenon;
                    else
                        throw std::invalid_argument(
                          "unsupported cavity gas " + gas);
                    Gases::CGas fill;
                    fill.addGasItem(1.0, gas_type);
                    model.materials().createGas(name, cavity_standard, fill);
                    model.materials().gas(name).setCavityConditions(
                      cavity_height, cavity_pressure);
                    if(!local_emissivities)
                    {
                        model.materials().gas(name).setSideEmissivities(
                          emissivity1, emissivity2);
                    }
                }
                material_names.emplace(material_key, name);
            }
        }

        for(std::size_t index = 0; index < nodes.size(); ++index)
        {
            const auto [id, x, y] = nodes[index];
            model.nodes().createNode(
              {.index = index + 1,
               .x = x,
               .y = y,
               .state = {.temperature = initial_temperature,
                          .humidity = initial_humidity,
                          .pressure = 101325.0,
                          .liquidPercent = 1.0}});
        }

        for(std::size_t index = 0; index < quads.size(); ++index)
        {
            const auto & quad = quads[index];
            const TransientMaterialKey material_key{
              materials[index], densities[index], heat_capacities[index]};
            model.createElement(
              {.node1 = lookup(quad[0]),
               .node2 = lookup(quad[1]),
               .node3 = lookup(quad[2]),
               .node4 = lookup(quad[3]),
               .material = material_names.at(material_key)});
        }

        std::set<Edge> cavity_surface_edges;
        for(const auto & [node1, node2, emissivity] :
            cavity_surface_emissivities)
        {
            const auto boundary = edge(node1, node2);
            if(node1 == node2 || !mesh_edges.contains(boundary))
            {
                throw std::invalid_argument(
                  "cavity surface emissivity must match a Q4 mesh edge");
            }
            if(!std::isfinite(emissivity) || emissivity < 0.0
               || emissivity > 1.0)
            {
                throw std::invalid_argument(
                  "cavity surface emissivity must be in [0, 1]");
            }
            if(!cavity_surface_edges
                  .insert(edge(lookup(node1), lookup(node2)))
                  .second)
            {
                throw std::invalid_argument(
                  "cavity surface emissivity edges must be unique");
            }
            model.setCavitySurfaceEmissivity(
              lookup(node1), lookup(node2), emissivity);
        }

        const auto check_boundary_edge = [&](const Edge & boundary,
                                             const char * what) {
            if(boundary.first == boundary.second
               || !mesh_edges.contains(boundary)
               || mesh_edges.at(boundary) != 1)
            {
                throw std::invalid_argument(
                  std::string(what)
                  + " boundary must match a Q4 exterior edge");
            }
        };

        bool anchored = false;
        // Diagnostics: slots allocated in the kernel's per-timestep BC
        // tables (see TransientSolution::bc_timestep_slots).
        std::size_t bc_slots = 0;
        std::set<Edge> claimed;
        for(const auto & [node1, node2, film_series, temperature_series] :
            convection)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "convection");
            check_series(film_series, "convection film");
            check_series(temperature_series,
                         "convection temperature");
            if(*std::min_element(film_series.begin(), film_series.end()) < 0.0)
            {
                throw std::invalid_argument(
                  "convection coefficients must be non-negative");
            }
            if(film_series.size() == 1 && temperature_series.size() == 1)
            {
                model.thermal().createBC_FixedHc(
                  lookup(node1), lookup(node2),
                  HygroThermFEM::FixedBCHCCoefficients(temperature_series[0],
                                                     film_series[0], 0.0),
                  false);
            }
            else
            {
                bc_slots += film_series.size() + temperature_series.size();
                model.thermal().createBC_PeriodicHc(
                  lookup(node1), lookup(node2),
                  HygroThermFEM::FixedHcPeriodicCoefficients{
                    .AirTemperatures = periodic(temperature_series),
                    .ConvectionCoefficients = periodic(film_series),
                    .AirHumidities = {}},
                  false);
            }
            if(*std::max_element(film_series.begin(), film_series.end()) > 0.0)
            {
                anchored = true;
            }
            claimed.insert(boundary);
        }
        for(const auto & [node1, node2, temperature_series] : temperatures)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "fixed temperature");
            check_series(temperature_series, "fixed temperature");
            if(temperature_series.size() == 1)
            {
                model.thermal().createBC_FixedTemperature(
                  lookup(node1), lookup(node2), temperature_series[0]);
            }
            else
            {
                bc_slots += temperature_series.size();
                model.thermal().createBC_PeriodicTemperature(
                  lookup(node1), lookup(node2),
                  periodic(temperature_series));
            }
            anchored = true;
            claimed.insert(boundary);
        }
        for(const auto & [node1, node2, flux_series] : fluxes)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "heat flux");
            check_series(flux_series, "heat flux");
            if(flux_series.size() == 1)
            {
                model.thermal().createBC_FixedFlux(lookup(node1), lookup(node2),
                                                   flux_series[0]);
            }
            else
            {
                bc_slots += flux_series.size();
                model.thermal().createBC_PeriodicFlux(
                  lookup(node1), lookup(node2), periodic(flux_series));
            }
            claimed.insert(boundary);
        }
        for(const auto & [node1, node2, emissivity_series,
                          temperature_series] : radiations)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "black-body radiation");
            check_series(emissivity_series,
                         "radiation emissivity");
            check_series(temperature_series,
                         "radiation temperature");
            if(*std::min_element(emissivity_series.begin(),
                                 emissivity_series.end())
                   < 0.0
               || *std::max_element(emissivity_series.begin(),
                                    emissivity_series.end())
                    > 1.0)
            {
                throw std::invalid_argument(
                  "radiation emissivities must be in [0, 1]");
            }
            if(emissivity_series.size() == 1 && temperature_series.size() == 1)
            {
                model.thermal().createBC_BlackBodyRadiation(
                  lookup(node1), lookup(node2), emissivity_series[0],
                  temperature_series[0]);
            }
            else
            {
                bc_slots += emissivity_series.size()
                            + temperature_series.size();
                model.thermal().createBC_PeriodicBlackBodyRadiation(
                  lookup(node1), lookup(node2),
                  periodic(emissivity_series),
                  periodic(temperature_series));
            }
            if(*std::max_element(emissivity_series.begin(),
                                 emissivity_series.end())
               > 0.0)
            {
                anchored = true;
            }
            claimed.insert(boundary);
        }
        for(const auto & [node1, node2, coefficient_series,
                          temperature_series] : linearized_radiations)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "linearized radiation");
            check_series(coefficient_series, "radiation hr");
            check_series(temperature_series,
                         "radiation temperature");
            if(*std::min_element(coefficient_series.begin(),
                                 coefficient_series.end())
               < 0.0)
            {
                throw std::invalid_argument(
                  "linearized radiation coefficients must be non-negative");
            }
            if(coefficient_series.size() == 1 && temperature_series.size() == 1)
            {
                model.thermal().createBC_LinearizedRadiation(
                  lookup(node1), lookup(node2),
                  HygroThermFEM::LinearizedRadiationBCCoefficients(
                    coefficient_series[0], temperature_series[0]));
            }
            else
            {
                bc_slots += coefficient_series.size()
                            + temperature_series.size();
                model.thermal().createBC_PeriodicLinearizedRadiation(
                  lookup(node1), lookup(node2),
                  periodic(coefficient_series),
                  periodic(temperature_series));
            }
            if(*std::max_element(coefficient_series.begin(),
                                 coefficient_series.end())
               > 0.0)
            {
                anchored = true;
            }
            claimed.insert(boundary);
        }
        if(!anchored)
        {
            throw std::invalid_argument(
              "transient model requires convection, radiation or fixed "
              "temperature to anchor temperature");
        }

        const auto frame_rates = [&](std::size_t step) {
            std::vector<TransientBoundaryEdgeRate> rates;
            for(const auto & rate :
                model.thermal().boundaryHeatRates(
                  HygroThermFEM::Variable::temperature, step))
            {
                // Kernel node indexes are the 1-based FEM indices; THMZ
                // callers key edges by 0-based node positions.
                rates.emplace_back(rate.node1 - 1, rate.node2 - 1,
                                   rate.heatRate);
            }
            return rates;
        };
        const auto frame_flux = [](const std::vector<HygroThermFEM::NodeFlux> &
                                     fluxes_in) {
            std::vector<std::array<double, 2>> values;
            values.reserve(fluxes_in.size());
            for(const auto & flux : fluxes_in)
            {
                if(!std::isfinite(flux.x) || !std::isfinite(flux.y))
                {
                    throw std::runtime_error(
                      "transient solve returned non-finite heat flux");
                }
                values.push_back({flux.x, flux.y});
            }
            return values;
        };
        // Per-edge interval exchange booked by the kernel accumulators over
        // the public step's accepted sub-steps. Keys are converted from
        // 1-based FEM indexes to the 0-based THMZ node positions.
        const auto frame_exchange =
          [](const std::map<std::pair<size_t, size_t>, double> & ledger) {
              std::vector<TransientBoundaryEdgeRate> exchanges;
              exchanges.reserve(ledger.size());
              for(const auto & [edge, exchange] : ledger)
              {
                  exchanges.emplace_back(edge.first - 1, edge.second - 1,
                                         exchange);
              }
              return exchanges;
          };

        // The t = 0 frame reflects the declared initial state: the conduction
        // flux of a uniform field is identically zero, while the boundary
        // exchange rates evaluate the timestep-0 conditions on the initial
        // temperatures (same convention as MultiDomain::currentStateSolution).
        if(on_step)
        {
            TransientFrame initial;
            initial.step = static_cast<std::size_t>(-1);
            initial.time = 0.0;
            initial.temperature =
              model.nodes().properties(HygroThermFEM::Variable::temperature);
            initial.humidity =
              model.nodes().properties(HygroThermFEM::Variable::humidity);
            initial.water_content =
              model.nodes().properties(HygroThermFEM::Variable::water);
            initial.ice_content.assign(nodes.size(), 0.0);
            initial.heat_flux.assign(nodes.size(), {0.0, 0.0});
            // Dry solve: no moisture domain, so the moisture columns are
            // identically zero rather than fabricated.
            initial.water_flux.assign(nodes.size(), {0.0, 0.0});
            initial.edge_heat_rates = frame_rates(0);
            // No interval precedes the t=0 frame: the exchange columns are
            // identically zero.
            on_step(initial);
        }

        std::vector<double> current_temperature(nodes.size(),
                                                initial_temperature);
        std::vector<double> current_humidity(nodes.size(), initial_humidity);
        TransientSolution result;
        result.bc_timestep_slots = bc_slots;
        for(std::size_t step = 0; step < num_steps; ++step)
        {
            const auto solution = model.transient(
              current_temperature, current_humidity, time_step, step);
            if(solution.temperature.size() != nodes.size()
               || !detail::finite(solution.temperature))
            {
                throw std::runtime_error(
                  "transient solve returned incomplete or non-finite results");
            }
            current_temperature = solution.temperature;
            current_humidity = solution.humidity;
            if(on_step)
            {
                TransientFrame frame;
                frame.step = step;
                frame.time = time_step * static_cast<double>(step + 1);
                frame.temperature = solution.temperature;
                frame.humidity = solution.humidity;
                frame.water_content = solution.waterContent;
                // Dry solve: no moisture domain; the ice and water-flux
                // columns stay identically zero.
                frame.ice_content.assign(nodes.size(), 0.0);
                frame.heat_flux = frame_flux(solution.heatFlux);
                frame.water_flux.assign(nodes.size(), {0.0, 0.0});
                frame.edge_heat_rates = frame_rates(step);
                frame.edge_heat_exchange =
                  frame_exchange(model.stepBoundaryHeatExchange());
                frame.temperature_error = solution.temperatureError;
                on_step(frame);
            }
            result.temperature = solution.temperature;
            result.heat_flux = frame_flux(solution.heatFlux);
            ++result.steps;
        }
        return result;
    }

    namespace
    {
        void check_curve(const std::vector<TablePoint> & curve,
                         const char * what,
                         bool humidity_keyed)
        {
            if(curve.empty())
            {
                throw std::invalid_argument(
                  std::string(what) + " table is required for hygrothermal "
                  "materials");
            }
            double previous = -std::numeric_limits<double>::infinity();
            for(const auto & [x, y] : curve)
            {
                if(!std::isfinite(x) || !std::isfinite(y) || y < 0.0
                   || x <= previous
                   || (humidity_keyed && (x < 0.0 || x > 1.0)))
                {
                    throw std::invalid_argument(
                      std::string(what) + " table must be finite with "
                      "strictly increasing in-range keys");
                }
                previous = x;
            }
        }

        std::vector<FenestrationCommon::point>
          to_points(const std::vector<TablePoint> & table)
        {
            std::vector<FenestrationCommon::point> points;
            points.reserve(table.size());
            for(const auto & [x, y] : table)
            {
                points.emplace_back(x, y);
            }
            return points;
        }
    }   // namespace

    TransientSolution solve_hygrothermal(
      const std::vector<NodeInput> & nodes,
      const std::vector<QuadInput> & quads,
      const std::vector<MaterialInput> & materials,
      const std::vector<std::optional<HygrothermalMaterialInput>> &
        hygro_materials,
      const PhysicsOptionsInput & physics,
      const std::vector<TimestepConvectionInput> & convection,
      const std::vector<TimestepHumidConvectionInput> & humid_convection,
      const std::vector<TimestepTemperatureInput> & temperatures,
      const std::vector<TimestepHumidityInput> & humidities,
      const std::vector<TimestepTemperatureHumidityInput> &
        temperature_humidities,
      const std::vector<TimestepFluxInput> & fluxes,
      const std::vector<TimestepRadiationInput> & radiations,
      const std::vector<TimestepLinearizedRadiationInput> &
        linearized_radiations,
      const double initial_temperature,
      const double initial_humidity,
      const double time_step,
      const std::size_t num_steps,
      const double convergence_tolerance,
      const double relaxation,
      const std::array<double, 3> & gravity,
      const std::vector<CavitySurfaceEmissivityInput> &
        cavity_surface_emissivities,
      const std::function<void(const TransientFrame &)> & on_step)
    {
        if(nodes.size() < 4 || quads.empty())
        {
            throw std::invalid_argument(
              "at least four nodes and one Q4 element are required");
        }
        if(materials.size() != quads.size()
           || hygro_materials.size() != quads.size())
        {
            throw std::invalid_argument(
              "one material and one hygrothermal record is required for each "
              "Q4 element");
        }
        if(!std::isfinite(initial_temperature)
           || !std::isfinite(initial_humidity) || initial_humidity < 0.0
           || initial_humidity > 1.0)
        {
            throw std::invalid_argument(
              "initial temperature must be finite and humidity in [0, 1]");
        }
        if(!std::isfinite(time_step) || time_step <= 0.0 || num_steps == 0)
        {
            throw std::invalid_argument(
              "time step must be positive and at least one step is required");
        }
        if(!std::isfinite(convergence_tolerance) || convergence_tolerance <= 0.0
           || !std::isfinite(relaxation) || relaxation <= 0.0
           || relaxation > 1.0)
        {
            throw std::invalid_argument(
              "convergence tolerance must be positive and relaxation in (0, 1]");
        }

        const auto fem_id = detail::femIndexMap(nodes);
        const auto lookup = [&](std::size_t id) {
            const auto found = fem_id.find(id);
            if(found == fem_id.end())
            {
                throw std::invalid_argument(
                  "element or boundary references an unknown node id");
            }
            return found->second;
        };

        const auto & [exclude_liquid, exclude_evaporation, exclude_capillary,
                      exclude_vapor_conduction, exclude_fusion,
                      variable_conductivity] = physics;

        HygroThermFEM::MultiDomain model(
          {.performThermal = true, .performMoisture = true});
        if(!detail::finite({gravity[0], gravity[1], gravity[2]})
           || std::abs(gravity[0] * gravity[0] + gravity[1] * gravity[1]
                       + gravity[2] * gravity[2] - 1.0) > 1e-12)
        {
            throw std::invalid_argument("gravity must be a finite unit vector");
        }
        model.setGravityVector({gravity[0], gravity[1], gravity[2]});
        model.setPhysicsOptions(
          {.excludeWaterLiquidTransportation = exclude_liquid,
           .excludeHeatOfEvaporation = exclude_evaporation,
           .excludeCapillaryConduction = exclude_capillary,
           .excludeVaporDiffusionConduction = exclude_vapor_conduction,
           .thermalConductivityMoistureAndTemperatureDependent =
             variable_conductivity,
           .excludeLatentHeatOfFusion = exclude_fusion});
        model.setSolverSettings({.relaxationParameter = relaxation,
                                  .errorTolerance = convergence_tolerance,
                                  .maxNumberOfIterations = 32,
                                  .maxDivisions = 8,
                                  .numberOfSubtimesteps = 8});

        const bool needs_liquid_curve = !exclude_liquid || !exclude_capillary;

        std::map<Edge, std::size_t> mesh_edges;
        using HygrothermalMaterialKey =
          std::tuple<MaterialInput, std::optional<HygrothermalMaterialInput>>;
        std::map<HygrothermalMaterialKey, std::string> material_names;
        for(std::size_t index = 0; index < quads.size(); ++index)
        {
            const auto & quad = quads[index];
            const auto & spec = materials[index];
            const auto & hygro = hygro_materials[index];
            const auto & [conductivity, emissivity1, emissivity2, standard,
                          gas, key, local_emissivities, cavity_height,
                          cavity_pressure] = spec;
            if(!std::isfinite(emissivity1) || emissivity1 < 0 || emissivity1 > 1
               || !std::isfinite(emissivity2) || emissivity2 < 0
               || emissivity2 > 1)
            {
                throw std::invalid_argument(
                  "material emissivities must be in [0, 1]");
            }
            std::set<std::size_t> distinct(quad.begin(), quad.end());
            if(distinct.size() != 4)
            {
                throw std::invalid_argument(
                  "each Q4 element must have four distinct node ids");
            }
            for(const auto id : quad)
            {
                (void)lookup(id);
            }
            for(std::size_t corner = 0; corner < quad.size(); ++corner)
            {
                ++mesh_edges[edge(quad[corner], quad[(corner + 1) % quad.size()])];
            }
            const HygrothermalMaterialKey material_key{spec, hygro};
            if(!material_names.contains(material_key))
            {
                const auto name =
                  "material " + std::to_string(material_names.size() + 1);
                if(standard.empty())
                {
                    if(!hygro.has_value())
                    {
                        throw std::invalid_argument(
                          "solid materials require hygrothermal properties");
                    }
                    const auto & [density, porosity, heat_capacity,
                                  diffusion_resistance, sorption, liquid,
                                  lambda_water, lambda_moisture_measured_temp,
                                  lambda_temperature,
                                  lambda_water_reference] = *hygro;
                    const auto invalid_property = [&](const char * field,
                                                       double received,
                                                       const char * range) {
                        std::ostringstream message;
                        message << name << ": " << field << " received "
                                << std::setprecision(
                                     std::numeric_limits<double>::max_digits10)
                                << received << "; " << range;
                        throw std::invalid_argument(message.str());
                    };
                    if(!std::isfinite(density) || density <= 0.0)
                    {
                        invalid_property("BulkDensity", density,
                                         "must be finite and positive");
                    }
                    if(!std::isfinite(porosity) || porosity <= 0.0
                       || porosity > 1.0)
                    {
                        invalid_property("Porosity", porosity,
                                         "must be finite and in (0, 1]");
                    }
                    if(!std::isfinite(heat_capacity) || heat_capacity <= 0.0)
                    {
                        invalid_property("SpecificHeatCapacityDry",
                                         heat_capacity,
                                         "must be finite and positive");
                    }
                    if(!std::isfinite(diffusion_resistance)
                       || diffusion_resistance <= 0.0)
                    {
                        invalid_property("WaterVaporDiffusionResistanceFactor",
                                         diffusion_resistance,
                                         "must be finite and positive");
                    }
                    check_curve(sorption, "sorption", true);
                    // The liquid transport curve is keyed by water content
                    // and must cover the sorption curve's saturation content;
                    // the kernel only prints a warning on mismatch, so the
                    // binding rejects instead of solving a silently degraded
                    // table.
                    if(needs_liquid_curve)
                    {
                        check_curve(liquid, "liquid transport", false);
                        const auto saturation = std::max_element(
                          sorption.begin(), sorption.end(),
                          [](const TablePoint & a, const TablePoint & b) {
                              return a.second < b.second;
                          })->second;
                        if(std::abs(liquid.back().first - saturation)
                           > 1e-9 * std::max(1.0, saturation))
                        {
                            throw std::invalid_argument(
                              "liquid transport table maximum water content "
                              "must equal the sorption curve saturation "
                              "content");
                        }
                    }
                    if(!variable_conductivity)
                    {
                        if(!std::isfinite(conductivity) || conductivity <= 0.0)
                        {
                            throw std::invalid_argument(
                              "conductivities must be finite and positive");
                        }
                    }
                    else
                    {
                        check_curve(lambda_water,
                                    "moisture dependent conductivity", false);
                        check_curve(lambda_temperature,
                                    "temperature dependent conductivity",
                                    false);
                        if(lambda_water.front().first < 0.0
                           || !std::isfinite(lambda_moisture_measured_temp)
                           || !std::isfinite(lambda_water_reference)
                           || lambda_water_reference < 0.0)
                        {
                            throw std::invalid_argument(
                              "conductivity table water content and measurement "
                              "references must be finite and non-negative");
                        }
                    }
                    auto & material =
                      model.materials().createSolidMaterial(name);
                    if(std::isfinite(conductivity) && conductivity > 0.0)
                    {
                        material.setThermalConductivity(conductivity);
                    }
                    material.setDensity(density);
                    material.setPorosity(porosity);
                    material.setHeatCapacity(heat_capacity);
                    material.setDiffusionResistanceFactor(
                      diffusion_resistance);
                    material.setSorptionCurve(to_points(sorption));
                    if(!liquid.empty())
                    {
                        material.setLiquidTransportationCurve(
                          to_points(liquid));
                    }
                    if(variable_conductivity)
                    {
                        // The 2D table constructor verifies both tables agree
                        // at each other's measurement reference (abs 1e-6).
                        material
                          .setThermalConductivityMoistureAndTemperatureDependent(
                            to_points(lambda_water),
                            lambda_moisture_measured_temp,
                            to_points(lambda_temperature),
                            lambda_water_reference);
                    }
                    material.setEmissivity(emissivity1);
                }
                else
                {
                    if(hygro.has_value())
                    {
                        throw std::invalid_argument(
                          "gas cavity materials must not carry hygrothermal "
                          "properties");
                    }
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
                        cavity_standard =
                          HygroThermFEM::CavityStandard::ISO15099Ventilated;
                    }
                    else
                    {
                        throw std::invalid_argument(
                          "unsupported cavity standard " + standard);
                    }
                    Gases::GasDef gas_type;
                    if(gas == "Air") gas_type = Gases::GasDef::Air;
                    else if(gas == "Argon") gas_type = Gases::GasDef::Argon;
                    else if(gas == "Krypton") gas_type = Gases::GasDef::Krypton;
                    else if(gas == "Xenon") gas_type = Gases::GasDef::Xenon;
                    else
                        throw std::invalid_argument(
                          "unsupported cavity gas " + gas);
                    Gases::CGas fill;
                    fill.addGasItem(1.0, gas_type);
                    model.materials().createGas(name, cavity_standard, fill);
                    model.materials().gas(name).setCavityConditions(
                      cavity_height, cavity_pressure);
                    if(!local_emissivities)
                    {
                        model.materials().gas(name).setSideEmissivities(
                          emissivity1, emissivity2);
                    }
                }
                material_names.emplace(material_key, name);
            }
        }

        for(std::size_t index = 0; index < nodes.size(); ++index)
        {
            const auto [id, x, y] = nodes[index];
            model.nodes().createNode(
              {.index = index + 1,
               .x = x,
               .y = y,
               .state = {.temperature = initial_temperature,
                          .humidity = initial_humidity,
                          .pressure = 101325.0,
                          .liquidPercent = 1.0}});
        }

        for(std::size_t index = 0; index < quads.size(); ++index)
        {
            const auto & quad = quads[index];
            const HygrothermalMaterialKey material_key{
              materials[index], hygro_materials[index]};
            model.createElement(
              {.node1 = lookup(quad[0]),
               .node2 = lookup(quad[1]),
               .node3 = lookup(quad[2]),
               .node4 = lookup(quad[3]),
               .material = material_names.at(material_key)});
        }

        std::set<Edge> cavity_surface_edges;
        for(const auto & [node1, node2, emissivity] :
            cavity_surface_emissivities)
        {
            const auto boundary = edge(node1, node2);
            if(node1 == node2 || !mesh_edges.contains(boundary))
            {
                throw std::invalid_argument(
                  "cavity surface emissivity must match a Q4 mesh edge");
            }
            if(!std::isfinite(emissivity) || emissivity < 0.0
               || emissivity > 1.0)
            {
                throw std::invalid_argument(
                  "cavity surface emissivity must be in [0, 1]");
            }
            if(!cavity_surface_edges
                  .insert(edge(lookup(node1), lookup(node2)))
                  .second)
            {
                throw std::invalid_argument(
                  "cavity surface emissivity edges must be unique");
            }
            model.setCavitySurfaceEmissivity(
              lookup(node1), lookup(node2), emissivity);
        }

        const auto check_boundary_edge = [&](const Edge & boundary,
                                             const char * what) {
            if(boundary.first == boundary.second
               || !mesh_edges.contains(boundary)
               || mesh_edges.at(boundary) != 1)
            {
                throw std::invalid_argument(
                  std::string(what)
                  + " boundary must match a Q4 exterior edge");
            }
        };
        const auto check_humidity_series = [](const std::vector<double> & series,
                                              const char * what) {
            for(const auto entry : series)
            {
                if(!std::isfinite(entry) || entry < 0.0 || entry > 1.0)
                {
                    throw std::invalid_argument(
                      std::string(what) + " humidity values must be in [0, 1]");
                }
            }
        };

        bool anchored = false;
        // Diagnostics: slots allocated in the kernel's per-timestep BC
        // tables (see TransientSolution::bc_timestep_slots).
        std::size_t bc_slots = 0;
        for(const auto & [node1, node2, film_series, temperature_series] :
            convection)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "convection");
            check_series(film_series, "convection film");
            check_series(temperature_series,
                         "convection temperature");
            if(*std::min_element(film_series.begin(), film_series.end()) < 0.0)
            {
                throw std::invalid_argument(
                  "convection coefficients must be non-negative");
            }
            // Dry convection on a wet model: no vapor exchange across this
            // edge, so the thermal BC drops the latent-heat term
            // (simulateVaporFluxEnergy = false).
            if(film_series.size() == 1 && temperature_series.size() == 1)
            {
                model.thermal().createBC_FixedHc(
                  lookup(node1), lookup(node2),
                  HygroThermFEM::FixedBCHCCoefficients(temperature_series[0],
                                                     film_series[0], 0.0),
                  false);
            }
            else
            {
                bc_slots += film_series.size() + temperature_series.size();
                model.thermal().createBC_PeriodicHc(
                  lookup(node1), lookup(node2),
                  HygroThermFEM::FixedHcPeriodicCoefficients{
                    .AirTemperatures = periodic(temperature_series),
                    .ConvectionCoefficients = periodic(film_series),
                    .AirHumidities = {}},
                  false);
            }
            if(*std::max_element(film_series.begin(), film_series.end()) > 0.0)
            {
                anchored = true;
            }
        }
        for(const auto & [node1, node2, film_series, temperature_series,
                          humidity_series] : humid_convection)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "humid convection");
            check_series(film_series, "humid convection film");
            check_series(temperature_series,
                         "humid convection temperature");
            check_series(humidity_series,
                         "humid convection humidity");
            check_humidity_series(humidity_series, "humid convection");
            if(*std::min_element(film_series.begin(), film_series.end()) < 0.0)
            {
                throw std::invalid_argument(
                  "convection coefficients must be non-negative");
            }
            // The MultiDomain overload attaches the thermal convection (with
            // the vapor-flux latent term) and the moisture Robin exchange
            // driven by the air humidity in one call.
            if(film_series.size() == 1 && temperature_series.size() == 1
               && humidity_series.size() == 1)
            {
                model.createBC_FixedHc(
                  lookup(node1), lookup(node2),
                  HygroThermFEM::FixedBCHCCoefficients(temperature_series[0],
                                                     film_series[0],
                                                     humidity_series[0]));
            }
            else
            {
                bc_slots += film_series.size() + temperature_series.size()
                            + humidity_series.size();
                model.createBC_PeriodicHc(
                  lookup(node1), lookup(node2),
                  HygroThermFEM::FixedHcPeriodicCoefficients{
                    .AirTemperatures = periodic(temperature_series),
                    .ConvectionCoefficients = periodic(film_series),
                    .AirHumidities = periodic(humidity_series)});
            }
            if(*std::max_element(film_series.begin(), film_series.end()) > 0.0)
            {
                anchored = true;
            }
        }
        for(const auto & [node1, node2, temperature_series] : temperatures)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "fixed temperature");
            check_series(temperature_series, "fixed temperature");
            if(temperature_series.size() == 1)
            {
                model.thermal().createBC_FixedTemperature(
                  lookup(node1), lookup(node2), temperature_series[0]);
            }
            else
            {
                bc_slots += temperature_series.size();
                model.thermal().createBC_PeriodicTemperature(
                  lookup(node1), lookup(node2),
                  periodic(temperature_series));
            }
            anchored = true;
        }
        for(const auto & [node1, node2, humidity_series] : humidities)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "fixed humidity");
            check_series(humidity_series, "fixed humidity");
            check_humidity_series(humidity_series, "fixed humidity");
            // The kernel pins phi itself via its node-temperature
            // R_Vector override; the Temperature member of the record is
            // not used by that path (documented in the contract).
            if(humidity_series.size() == 1)
            {
                model.moisture().createBC_FixedHumidity(
                  lookup(node1), lookup(node2),
                  HygroThermFEM::TemperatureAndHumidity(0.0,
                                                        humidity_series[0]));
            }
            else
            {
                bc_slots += humidity_series.size();
                model.moisture().createBC_PeriodicHumidity(
                  lookup(node1), lookup(node2),
                  periodic(humidity_series));
            }
        }
        for(const auto & [node1, node2, temperature_series, humidity_series] :
            temperature_humidities)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "fixed temperature and humidity");
            check_series(temperature_series,
                         "fixed temperature and humidity temperature");
            check_series(humidity_series,
                         "fixed temperature and humidity humidity");
            check_humidity_series(humidity_series,
                                  "fixed temperature and humidity");
            if(temperature_series.size() == 1 && humidity_series.size() == 1)
            {
                model.createBC_FixedTemperatureAndHumidity(
                  lookup(node1), lookup(node2),
                  HygroThermFEM::TemperatureAndHumidity(temperature_series[0],
                                                        humidity_series[0]));
            }
            else
            {
                bc_slots += temperature_series.size()
                            + humidity_series.size();
                model.createBC_PeriodicTemperatureAndHumidity(
                  lookup(node1), lookup(node2),
                  periodic(temperature_series), periodic(humidity_series));
            }
            anchored = true;
        }
        for(const auto & [node1, node2, flux_series] : fluxes)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "heat flux");
            check_series(flux_series, "heat flux");
            if(flux_series.size() == 1)
            {
                model.thermal().createBC_FixedFlux(lookup(node1), lookup(node2),
                                                   flux_series[0]);
            }
            else
            {
                bc_slots += flux_series.size();
                model.thermal().createBC_PeriodicFlux(
                  lookup(node1), lookup(node2), periodic(flux_series));
            }
        }
        for(const auto & [node1, node2, emissivity_series,
                          temperature_series] : radiations)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "black-body radiation");
            check_series(emissivity_series,
                         "radiation emissivity");
            check_series(temperature_series,
                         "radiation temperature");
            if(*std::min_element(emissivity_series.begin(),
                                 emissivity_series.end())
                   < 0.0
               || *std::max_element(emissivity_series.begin(),
                                    emissivity_series.end())
                    > 1.0)
            {
                throw std::invalid_argument(
                  "radiation emissivities must be in [0, 1]");
            }
            if(emissivity_series.size() == 1 && temperature_series.size() == 1)
            {
                model.thermal().createBC_BlackBodyRadiation(
                  lookup(node1), lookup(node2), emissivity_series[0],
                  temperature_series[0]);
            }
            else
            {
                bc_slots += emissivity_series.size()
                            + temperature_series.size();
                model.thermal().createBC_PeriodicBlackBodyRadiation(
                  lookup(node1), lookup(node2),
                  periodic(emissivity_series),
                  periodic(temperature_series));
            }
            if(*std::max_element(emissivity_series.begin(),
                                 emissivity_series.end())
                > 0.0)
            {
                anchored = true;
            }
        }
        for(const auto & [node1, node2, coefficient_series,
                          temperature_series] : linearized_radiations)
        {
            const auto boundary = edge(node1, node2);
            check_boundary_edge(boundary, "linearized radiation");
            check_series(coefficient_series, "radiation hr");
            check_series(temperature_series,
                         "radiation temperature");
            if(*std::min_element(coefficient_series.begin(),
                                 coefficient_series.end())
                < 0.0)
            {
                throw std::invalid_argument(
                  "linearized radiation coefficients must be non-negative");
            }
            if(coefficient_series.size() == 1 && temperature_series.size() == 1)
            {
                model.thermal().createBC_LinearizedRadiation(
                  lookup(node1), lookup(node2),
                  HygroThermFEM::LinearizedRadiationBCCoefficients(
                    coefficient_series[0], temperature_series[0]));
            }
            else
            {
                bc_slots += coefficient_series.size()
                            + temperature_series.size();
                model.thermal().createBC_PeriodicLinearizedRadiation(
                  lookup(node1), lookup(node2),
                  periodic(coefficient_series),
                  periodic(temperature_series));
            }
            if(*std::max_element(coefficient_series.begin(),
                                 coefficient_series.end())
                > 0.0)
            {
                anchored = true;
            }
        }
        if(!anchored)
        {
            throw std::invalid_argument(
              "transient model requires convection, radiation or fixed "
              "temperature to anchor temperature");
        }

        const auto frame_rates = [&](const HygroThermFEM::Variable variable,
                                     std::size_t step) {
            const auto rates_raw =
              variable == HygroThermFEM::Variable::humidity
                ? model.moisture().boundaryHeatRates(variable, step)
                : model.thermal().boundaryHeatRates(variable, step);
            std::vector<TransientBoundaryEdgeRate> rates;
            for(const auto & rate : rates_raw)
            {
                // Kernel node indexes are the 1-based FEM indices; THMZ
                // callers key edges by 0-based node positions.
                rates.emplace_back(rate.node1 - 1, rate.node2 - 1,
                                   rate.heatRate);
            }
            return rates;
        };
        const auto frame_flux = [](const std::vector<HygroThermFEM::NodeFlux> &
                                     fluxes_in,
                                   const char * what) {
            std::vector<std::array<double, 2>> values;
            values.reserve(fluxes_in.size());
            for(const auto & flux : fluxes_in)
            {
                if(!std::isfinite(flux.x) || !std::isfinite(flux.y))
                {
                    throw std::runtime_error(
                      std::string("transient solve returned non-finite ")
                      + what);
                }
                values.push_back({flux.x, flux.y});
            }
            return values;
        };
        // Per-edge interval exchange booked by the kernel accumulators over
        // the public step's accepted sub-steps. Keys are converted from
        // 1-based FEM indexes to the 0-based THMZ node positions.
        const auto frame_exchange =
          [](const std::map<std::pair<size_t, size_t>, double> & ledger) {
              std::vector<TransientBoundaryEdgeRate> exchanges;
              exchanges.reserve(ledger.size());
              for(const auto & [edge, exchange] : ledger)
              {
                  exchanges.emplace_back(edge.first - 1, edge.second - 1,
                                         exchange);
              }
              return exchanges;
          };

        if(on_step)
        {
            TransientFrame initial;
            initial.step = static_cast<std::size_t>(-1);
            initial.time = 0.0;
            initial.temperature =
              model.nodes().properties(HygroThermFEM::Variable::temperature);
            initial.humidity =
              model.nodes().properties(HygroThermFEM::Variable::humidity);
            initial.water_content =
              model.nodes().properties(HygroThermFEM::Variable::water);
            initial.ice_content =
              model.nodes().properties(HygroThermFEM::Variable::ice);
            initial.heat_flux = frame_flux(model.thermal().flux(), "heat flux");
            initial.water_flux = frame_flux(model.moisture().flux(), "water flux");
            initial.edge_heat_rates =
              frame_rates(HygroThermFEM::Variable::temperature, 0);
            initial.edge_water_rates =
              frame_rates(HygroThermFEM::Variable::humidity, 0);
            // No interval precedes the t=0 frame: the exchange columns are
            // identically zero.
            on_step(initial);
        }

        std::vector<double> current_temperature(nodes.size(),
                                                initial_temperature);
        std::vector<double> current_humidity(nodes.size(), initial_humidity);
        TransientSolution result;
        result.bc_timestep_slots = bc_slots;
        for(std::size_t step = 0; step < num_steps; ++step)
        {
            const auto solution = model.transient(
              current_temperature, current_humidity, time_step, step);
            if(solution.temperature.size() != nodes.size()
               || solution.humidity.size() != nodes.size()
               || !detail::finite(solution.temperature)
               || !detail::finite(solution.humidity))
            {
                throw std::runtime_error(
                  "hygrothermal solve returned incomplete or non-finite "
                  "results");
            }
            current_temperature = solution.temperature;
            current_humidity = solution.humidity;
            if(on_step)
            {
                TransientFrame frame;
                frame.step = step;
                frame.time = time_step * static_cast<double>(step + 1);
                frame.temperature = solution.temperature;
                frame.humidity = solution.humidity;
                frame.water_content = solution.waterContent;
                frame.ice_content = solution.iceContent;
                frame.heat_flux = frame_flux(solution.heatFlux, "heat flux");
                frame.water_flux = frame_flux(solution.waterFlux,
                                              "water flux");
                frame.edge_heat_rates =
                  frame_rates(HygroThermFEM::Variable::temperature, step);
                frame.edge_water_rates =
                  frame_rates(HygroThermFEM::Variable::humidity, step);
                frame.edge_heat_exchange =
                  frame_exchange(model.stepBoundaryHeatExchange());
                frame.edge_water_exchange =
                  frame_exchange(model.stepBoundaryWaterExchange());
                frame.temperature_error = solution.temperatureError;
                frame.humidity_error = solution.humidityError;
                on_step(frame);
            }
            result.temperature = solution.temperature;
            result.humidity = solution.humidity;
            result.heat_flux = frame_flux(solution.heatFlux, "heat flux");
            result.water_flux = frame_flux(solution.waterFlux, "water flux");
            ++result.steps;
        }
        return result;
    }
}   // namespace pyhygrotherm
