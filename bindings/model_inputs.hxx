#pragma once

// Shared THMZ-side input helpers for the flat pyhygrotherm bindings.
// Both solvers map user node ids onto the kernel's 1-based FEM indices in
// the same way: node positions are THMZ sequence positions (0-based) and
// kernel indices are position + 1.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace pyhygrotherm
{
    namespace detail
    {
        using Edge = std::pair<std::size_t, std::size_t>;

        inline Edge edge(std::size_t first, std::size_t second)
        {
            return std::minmax(first, second);
        }

        inline bool finite(const std::vector<double> & values)
        {
            return std::all_of(values.begin(), values.end(), [](double value) {
                return std::isfinite(value);
            });
        }

        template<typename Nodes>
        std::unordered_map<std::size_t, std::size_t> femIndexMap(
          const Nodes & nodes)
        {
            std::unordered_map<std::size_t, std::size_t> fem_id;
            fem_id.reserve(nodes.size());
            for(std::size_t index = 0; index < nodes.size(); ++index)
            {
                const auto [id, x, y] = nodes[index];
                if(!std::isfinite(x) || !std::isfinite(y))
                {
                    throw std::invalid_argument("node coordinates must be finite");
                }
                if(!fem_id.emplace(id, index + 1).second)
                {
                    throw std::invalid_argument("node ids must be unique");
                }
            }
            return fem_id;
        }
    }   // namespace detail
}   // namespace pyhygrotherm
