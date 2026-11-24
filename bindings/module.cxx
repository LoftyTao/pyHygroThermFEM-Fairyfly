#include "build_source.hxx"
#include "steady_solver.hxx"
#include "transient_solver.hxx"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>

PYBIND11_MODULE(_native, module)
{
    module.def("_build_source_identity", []() {
        return std::string(pyhygrotherm::kBuildSourceIdentity);
    });

    module.def(
      "solve_steady",
      [](const std::vector<pyhygrotherm::NodeInput> & nodes,
         const std::vector<pyhygrotherm::QuadInput> & quads,
         const pybind11::object & conductivity_input,
         const std::vector<pyhygrotherm::ConvectionInput> & convection,
         const std::vector<pyhygrotherm::RadiationInput> & radiations,
         const std::vector<pyhygrotherm::FluxInput> & fluxes,
         const std::vector<pyhygrotherm::TemperatureInput> & temperatures,
         const std::array<double, 3> & gravity,
         bool estimate_error,
         const std::vector<pyhygrotherm::CavitySurfaceEmissivityInput> &
           cavity_surface_emissivities) {
          if(PyBool_Check(conductivity_input.ptr()))
          {
              throw pybind11::value_error("conductivity must be a positive number or a sequence");
          }
          std::vector<pyhygrotherm::MaterialInput> materials;
          try
          {
              const auto value = pybind11::cast<double>(conductivity_input);
              materials.assign(quads.size(), {value, 0.9, 0.9, "", "", "",
                                              false, 1.0, 101325.0});
          }
          catch(const pybind11::cast_error &)
          {
              try
              {
                  for(const auto value : pybind11::cast<std::vector<double>>(conductivity_input))
                  {
                      materials.emplace_back(value, 0.9, 0.9, "", "", "",
                                             false, 1.0, 101325.0);
                  }
              }
              catch(const pybind11::cast_error &)
              {
                  materials = pybind11::cast<std::vector<pyhygrotherm::MaterialInput>>(
                    conductivity_input);
              }
          }
          auto solution = pyhygrotherm::solve_steady(
            nodes, quads, materials, convection, radiations, fluxes,
            temperatures, gravity,
            estimate_error, cavity_surface_emissivities);
          return pybind11::make_tuple(std::move(solution.temperature),
                                      std::move(solution.heat_flux),
                                      solution.energy_error_percent);
      },
      pybind11::arg("nodes"),
      pybind11::arg("quads"),
      pybind11::arg("conductivity"),
      pybind11::arg("convection"),
      pybind11::arg("radiations") = std::vector<pyhygrotherm::RadiationInput>{},
      pybind11::arg("fluxes") = std::vector<pyhygrotherm::FluxInput>{},
      pybind11::arg("temperatures") = std::vector<pyhygrotherm::TemperatureInput>{},
      pybind11::arg("gravity") = std::array<double, 3>{0.0, -1.0, 0.0},
      pybind11::arg("estimate_error") = false,
      pybind11::arg("cavity_surface_emissivities") =
        std::vector<pyhygrotherm::CavitySurfaceEmissivityInput>{},
      "Solve a dry, steady-state Q4 thermal model.");

    module.def(
      "solve_transient",
      [](const std::vector<pyhygrotherm::NodeInput> & nodes,
         const std::vector<pyhygrotherm::QuadInput> & quads,
         const std::vector<pyhygrotherm::MaterialInput> & materials,
         const std::vector<double> & densities,
         const std::vector<double> & heat_capacities,
         const std::vector<pyhygrotherm::TimestepConvectionInput> & convection,
         const std::vector<pyhygrotherm::TimestepTemperatureInput> &
           temperatures,
         const std::vector<pyhygrotherm::TimestepFluxInput> & fluxes,
         const std::vector<pyhygrotherm::TimestepRadiationInput> & radiations,
         const std::vector<pyhygrotherm::TimestepLinearizedRadiationInput> &
           linearized_radiations,
         double initial_temperature,
         double initial_humidity,
         double time_step,
         std::size_t num_steps,
         double convergence_tolerance,
         double relaxation,
         const std::array<double, 3> & gravity,
         const std::vector<pyhygrotherm::CavitySurfaceEmissivityInput> &
           cavity_surface_emissivities,
         const pybind11::object & on_step) {
          std::function<void(const pyhygrotherm::TransientFrame &)>
            step_handler;
          if(!on_step.is_none())
          {
              if(!PyCallable_Check(on_step.ptr()))
              {
                  throw pybind11::type_error(
                    "on_step must be a callable or None");
              }
              step_handler = [on_step](
                const pyhygrotherm::TransientFrame & frame) {
                  on_step(pybind11::make_tuple(
                    frame.step == static_cast<std::size_t>(-1)
                      ? -1
                      : static_cast<long long>(frame.step),
                    frame.time,
                    frame.temperature,
                    frame.humidity,
                    frame.water_content,
                    frame.ice_content,
                    frame.heat_flux,
                    frame.water_flux,
                    frame.edge_heat_rates,
                    frame.edge_water_rates,
                    frame.edge_heat_exchange,
                    frame.edge_water_exchange,
                    frame.temperature_error,
                    frame.humidity_error));
              };
          }
          auto solution = pyhygrotherm::solve_transient(
            nodes, quads, materials, densities, heat_capacities, convection,
            temperatures, fluxes, radiations, linearized_radiations,
            initial_temperature, initial_humidity, time_step, num_steps,
            convergence_tolerance, relaxation, gravity,
            cavity_surface_emissivities, step_handler);
          return pybind11::make_tuple(std::move(solution.temperature),
                                      std::move(solution.heat_flux),
                                      solution.steps,
                                      solution.bc_timestep_slots);
      },
      pybind11::arg("nodes"),
      pybind11::arg("quads"),
      pybind11::arg("materials"),
      pybind11::arg("densities"),
      pybind11::arg("heat_capacities"),
      pybind11::arg("convection") =
        std::vector<pyhygrotherm::TimestepConvectionInput>{},
      pybind11::arg("temperatures") =
        std::vector<pyhygrotherm::TimestepTemperatureInput>{},
      pybind11::arg("fluxes") =
        std::vector<pyhygrotherm::TimestepFluxInput>{},
      pybind11::arg("radiations") =
        std::vector<pyhygrotherm::TimestepRadiationInput>{},
      pybind11::arg("linearized_radiations") =
        std::vector<pyhygrotherm::TimestepLinearizedRadiationInput>{},
      pybind11::arg("initial_temperature") = 0.0,
      pybind11::arg("initial_humidity") = 0.0,
      pybind11::arg("time_step") = 3600.0,
      pybind11::arg("num_steps") = 1,
      pybind11::arg("convergence_tolerance") = 1e-5,
      pybind11::arg("relaxation") = 1.0,
      pybind11::arg("gravity") = std::array<double, 3>{0.0, -1.0, 0.0},
      pybind11::arg("cavity_surface_emissivities") =
        std::vector<pyhygrotherm::CavitySurfaceEmissivityInput>{},
      pybind11::arg("on_step") = pybind11::none(),
      R"doc(Solve a dry Q4 thermal transient and optionally stream each frame.

This internal binding does not validate the THMZ archive contract. External
callers should use pyhygrothermfem_fairyfly.solve or run_thmz. Each coefficient
field is constant for a one-value series, otherwise cycles by its own period;
fields are never combined into a least-common-multiple table.

Args:
    nodes: (node_id, x, y) tuples, with unique positive IDs and coordinates
        in metres. Node order determines all returned node arrays.
    quads: Counterclockwise four-node elements referencing node_id values.
    materials: One tuple per quad: (conductivity, emissivity1, emissivity2,
        cavity_standard, gas, region_key, local_emissivities, cavity_height,
        cavity_pressure). Conductivity is W/(m.K); emissivities are in [0, 1];
        height is metres and pressure is Pa. An empty cavity_standard denotes
        a solid; cavity names are CEN, ISO15099 or ISO15099Ventilated, and gas
        names are Air, Argon, Krypton or Xenon. region_key separates regions.
    densities: One solid density per quad in kg/m3, finite and positive.
        Entries for gas-cavity quads are not used.
    heat_capacities: One solid heat capacity per quad in J/(kg.K), finite and
        positive. Entries for gas-cavity quads are not used.
    convection: (node1, node2, film_series, air_temperature_series) exterior
        edges; film coefficients are non-negative W/(m2.K), temperatures C.
    temperatures: (node1, node2, temperature_series) exterior temperature
        pins in C. Pin nodes in the initial frame use the series first value.
    fluxes: (node1, node2, flux_series) exterior fluxes in W/m2, positive
        into the model.
    radiations: (node1, node2, emissivity_series, temperature_series) black
        body exchange edges; emissivities are in [0, 1], temperatures C.
    linearized_radiations: (node1, node2, hr_series, temperature_series)
        exterior edges; non-negative hr is W/(m2.K), temperatures C.
    initial_temperature: Uniform initial temperature in C before pin seeding.
    initial_humidity: Relative humidity in [0, 1], retained as a dry placeholder;
        no moisture transport is solved.
    time_step: Positive timestep duration in seconds.
    num_steps: Positive number of completed timesteps to solve.
    convergence_tolerance: Positive, dimensionless kernel convergence tolerance.
    relaxation: Dimensionless relaxation factor in (0, 1].
    gravity: Finite unit direction vector, default (0, -1, 0).
    cavity_surface_emissivities: (node1, node2, emissivity) mesh-edge overrides
        for local cavity surfaces; emissivity is in [0, 1].
    on_step: Callable receiving one fourteen-item tuple in this exact order:
        (step, time, temperature, humidity, water_content, ice_content,
        heat_flux, water_flux, edge_heat_rates, edge_water_rates,
        edge_heat_exchange, edge_water_exchange, temperature_error,
        humidity_error). step is -1 for the t=0 frame,
        then 0..num_steps-1; time is seconds. Node arrays follow nodes order:
        temperatures are C, humidity is relative humidity, water and ice
        content are kg/m3, heat_flux is (x, y) W/m2, water_flux is (x, y)
        kg/(m2.s). Edge-rate entries are (node1_position, node2_position, rate),
        using zero-based positions, not node_id; heat rates are W/m and water
        rates kg/(m.s), outward-positive per metre of depth. Exchange entries
        share the same key shape but carry the interval quantity accumulated
        over the accepted sub-steps ending at that frame: J/m for heat, kg/m
        for water; the t=0 frame carries zeros because no interval precedes
        it. The last two items
        are dimensionless kernel error measures. Dry moisture/ice/flux/rate
        fields are placeholders; no moisture physics is inferred from them.
        The initial heat_flux is zero. The callback runs within the solve;
        exceptions propagate immediately. None disables frame delivery.
        A consumer that retains frames forfeits bounded-memory streaming.

Returns:
    (temperatures, heat_fluxes, steps_completed, bc_timestep_slots). Final node
    temperatures are C and heat_fluxes are (x, y) W/m2 in nodes order.
    bc_timestep_slots counts the sum of periodic field lengths, not completed
    steps; it is only a BC-storage diagnostic, not a total-memory measurement.

Raises:
    ValueError: Invalid model arrays, coefficients or solver parameters.
    TypeError: on_step is neither callable nor None.
    RuntimeError: The kernel fails or returns incomplete/non-finite results.
    Exception: Any exception raised by on_step, propagated unchanged.
)doc");

    module.def(
      "solve_hygrothermal",
      [](const std::vector<pyhygrotherm::NodeInput> & nodes,
         const std::vector<pyhygrotherm::QuadInput> & quads,
         const std::vector<pyhygrotherm::MaterialInput> & materials,
         const std::vector<
           std::optional<pyhygrotherm::HygrothermalMaterialInput>> &
           hygro_materials,
         const pyhygrotherm::PhysicsOptionsInput & physics,
         const std::vector<pyhygrotherm::TimestepConvectionInput> &
           convection,
         const std::vector<pyhygrotherm::TimestepHumidConvectionInput> &
           humid_convection,
         const std::vector<pyhygrotherm::TimestepTemperatureInput> &
           temperatures,
         const std::vector<pyhygrotherm::TimestepHumidityInput> & humidities,
         const std::vector<pyhygrotherm::TimestepTemperatureHumidityInput> &
           temperature_humidities,
         const std::vector<pyhygrotherm::TimestepFluxInput> & fluxes,
         const std::vector<pyhygrotherm::TimestepRadiationInput> & radiations,
         const std::vector<pyhygrotherm::TimestepLinearizedRadiationInput> &
           linearized_radiations,
         double initial_temperature,
         double initial_humidity,
         double time_step,
         std::size_t num_steps,
         double convergence_tolerance,
         double relaxation,
         const std::array<double, 3> & gravity,
         const std::vector<pyhygrotherm::CavitySurfaceEmissivityInput> &
           cavity_surface_emissivities,
         const pybind11::object & on_step) {
          std::function<void(const pyhygrotherm::TransientFrame &)>
            step_handler;
          if(!on_step.is_none())
          {
              if(!PyCallable_Check(on_step.ptr()))
              {
                  throw pybind11::type_error(
                    "on_step must be a callable or None");
              }
              step_handler = [on_step](
                const pyhygrotherm::TransientFrame & frame) {
                  on_step(pybind11::make_tuple(
                    frame.step == static_cast<std::size_t>(-1)
                      ? -1
                      : static_cast<long long>(frame.step),
                    frame.time,
                    frame.temperature,
                    frame.humidity,
                    frame.water_content,
                    frame.ice_content,
                    frame.heat_flux,
                    frame.water_flux,
                    frame.edge_heat_rates,
                    frame.edge_water_rates,
                    frame.edge_heat_exchange,
                    frame.edge_water_exchange,
                    frame.temperature_error,
                    frame.humidity_error));
              };
          }
          auto solution = pyhygrotherm::solve_hygrothermal(
            nodes, quads, materials, hygro_materials, physics, convection,
            humid_convection, temperatures, humidities, temperature_humidities,
            fluxes, radiations, linearized_radiations,
            initial_temperature, initial_humidity, time_step, num_steps,
            convergence_tolerance, relaxation, gravity,
            cavity_surface_emissivities, step_handler);
          return pybind11::make_tuple(std::move(solution.temperature),
                                      std::move(solution.heat_flux),
                                      std::move(solution.humidity),
                                      std::move(solution.water_flux),
                                      solution.steps,
                                      solution.bc_timestep_slots);
      },
      pybind11::arg("nodes"),
      pybind11::arg("quads"),
      pybind11::arg("materials"),
      pybind11::arg("hygro_materials"),
      pybind11::arg("physics"),
      pybind11::arg("convection") =
        std::vector<pyhygrotherm::TimestepConvectionInput>{},
      pybind11::arg("humid_convection") =
        std::vector<pyhygrotherm::TimestepHumidConvectionInput>{},
      pybind11::arg("temperatures") =
        std::vector<pyhygrotherm::TimestepTemperatureInput>{},
      pybind11::arg("humidities") =
        std::vector<pyhygrotherm::TimestepHumidityInput>{},
      pybind11::arg("temperature_humidities") =
        std::vector<pyhygrotherm::TimestepTemperatureHumidityInput>{},
      pybind11::arg("fluxes") =
        std::vector<pyhygrotherm::TimestepFluxInput>{},
      pybind11::arg("radiations") =
        std::vector<pyhygrotherm::TimestepRadiationInput>{},
      pybind11::arg("linearized_radiations") =
        std::vector<pyhygrotherm::TimestepLinearizedRadiationInput>{},
      pybind11::arg("initial_temperature") = 0.0,
      pybind11::arg("initial_humidity") = 0.0,
      pybind11::arg("time_step") = 3600.0,
      pybind11::arg("num_steps") = 1,
      pybind11::arg("convergence_tolerance") = 1e-5,
      pybind11::arg("relaxation") = 1.0,
      pybind11::arg("gravity") = std::array<double, 3>{0.0, -1.0, 0.0},
      pybind11::arg("cavity_surface_emissivities") =
        std::vector<pyhygrotherm::CavitySurfaceEmissivityInput>{},
      pybind11::arg("on_step") = pybind11::none(),
      "Solve a coupled hygrothermal transient Q4 model, streaming each "
      "timestep to the on_step callback.");
}
