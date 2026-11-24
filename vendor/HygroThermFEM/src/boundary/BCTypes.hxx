#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace HygroThermFEM
{
    //! Number of nodes on a linear boundary segment.
    inline constexpr std::size_t numOfBCNodes = 2;

    //! \brief Cyclic coefficient series for a periodic boundary condition.
    //!
    //! One field of a time-varying boundary condition (e.g. the air
    //! temperature of a convection pair). The series repeats with its own
    //! period: value at timestep t is values[t % values.size()]. Because
    //! each field cycles independently, multi-field boundary conditions
    //! never need the combined lcm-period coefficient tables that would
    //! explode memory on coprime periods.
    struct PeriodicSeries
    {
        //! Value at a timestep, cyclically indexed; empty series yield 0 so
        //! optional fields (e.g. air humidity on a dry convection) may be
        //! left unpopulated.
        [[nodiscard]] double at(const std::size_t timestep) const
        {
            return values.empty() ? 0.0 : values[timestep % values.size()];
        }

        [[nodiscard]] std::size_t period() const
        {
            return values.size();
        }

        std::vector<double> values;
    };

    //! \brief Nodal values along a two-node boundary segment, kept on the stack.
    //!
    //! Every boundary condition is re-evaluated once per Newton-Raphson iteration, so
    //! holding these two values in a heap vector meant several allocations per segment
    //! per iteration across the R-vector and H-matrix coefficient chains.
    using BCVector = std::array<double, numOfBCNodes>;

    [[nodiscard]] inline BCVector operator+(const BCVector & first, const BCVector & second)
    {
        BCVector result{};
        for(std::size_t idx = 0; idx < numOfBCNodes; ++idx)
        {
            result[idx] = first[idx] + second[idx];
        }
        return result;
    }

    //! Element-by-element product.
    [[nodiscard]] inline BCVector operator*(const BCVector & first, const BCVector & second)
    {
        BCVector result{};
        for(std::size_t idx = 0; idx < numOfBCNodes; ++idx)
        {
            result[idx] = first[idx] * second[idx];
        }
        return result;
    }

    [[nodiscard]] inline BCVector operator*(const BCVector & first, const double scalar)
    {
        BCVector result{};
        for(std::size_t idx = 0; idx < numOfBCNodes; ++idx)
        {
            result[idx] = first[idx] * scalar;
        }
        return result;
    }

    [[nodiscard]] inline BCVector operator*(const double scalar, const BCVector & second)
    {
        return second * scalar;
    }

    //! \brief Dense, inline 2x2 matrix for boundary-condition assembly.
    //!
    //! The H-matrix of a two-node boundary segment is fully populated, so the previous
    //! sparse SquareMatrix representation paid Eigen sparse construction and insertion
    //! for four values on every evaluation.
    struct BCMatrix2D
    {
        //! Matrix dimension; mirrors SquareMatrix::size() for drop-in use.
        [[nodiscard]] static constexpr std::size_t size()
        {
            return numOfBCNodes;
        }

        double & operator()(const std::size_t row, const std::size_t col)
        {
            return values[row * numOfBCNodes + col];
        }

        double operator()(const std::size_t row, const std::size_t col) const
        {
            return values[row * numOfBCNodes + col];
        }

        //! Every column scaled by the matching nodal coefficient. Matches the behavior of
        //! SquareMatrix::mmultRows (matrix * diag(coefficients)), which this replaces on the
        //! boundary-condition path.
        [[nodiscard]] BCMatrix2D mmultColumns(const BCVector & coefficients) const
        {
            BCMatrix2D result{};
            for(std::size_t row = 0; row < numOfBCNodes; ++row)
            {
                for(std::size_t col = 0; col < numOfBCNodes; ++col)
                {
                    result(row, col) = (*this)(row, col) * coefficients[col];
                }
            }
            return result;
        }

        std::array<double, numOfBCNodes * numOfBCNodes> values{};
    };
}   // namespace HygroThermFEM
