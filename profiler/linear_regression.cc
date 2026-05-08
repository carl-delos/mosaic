// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "linear_regression.h"

#include <numeric>  // For std::accumulate
#include <set>      // For std::set

/**
 * @brief Construct a LinearRegression accumulator.
 *
 * @param[in] mode Regression mode controlling how points are retained.
 */
LinearRegression::LinearRegression(Mode mode) : mode_(mode), sumX(0.0), sumY(0.0), sumXY(0.0), sumX2(0.0), n(0) {}

/**
 * @brief Add a transfer sample to the regression state.
 *
 * @param[in] x Transfer size in bytes.
 * @param[in] y Transfer time in microseconds.
 */
void LinearRegression::addPoint(double x, double y)
{
    if (mode_ == Mode::AVG)
    {
        // Original behavior: use all points
        dataPoints.push_back({x, y});
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
        n++;
    }
    else if (mode_ == Mode::MIN)
    {
        // MIN mode: track minimum time for each size
        auto it = minTimesPerSize.find(x);
        if (it == minTimesPerSize.end())
        {
            // First time seeing this size
            minTimesPerSize[x] = y;
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
            n++;
        }
        else if (y < it->second)
        {
            // New minimum time for this size - update the point
            double oldY = it->second;
            it->second  = y;

            // Update sums: subtract old contribution, add new
            sumY  = sumY - oldY + y;
            sumXY = sumXY - x * oldY + x * y;
            // sumX and sumX2 don't change since x is the same
        }
        // If y >= current min, do nothing (keep existing minimum)
    }
}

/**
 * @brief Merge samples from another accumulator.
 *
 * @param[in] other Source accumulator to merge into this one.
 */
void LinearRegression::merge(const LinearRegression& other)
{
    if (mode_ == Mode::AVG)
    {
        // Original behavior: merge all statistics
        sumX += other.sumX;
        sumY += other.sumY;
        sumXY += other.sumXY;
        sumX2 += other.sumX2;
        n += other.n;

        // Merge data points (for potential future use)
        dataPoints.insert(dataPoints.end(), other.dataPoints.begin(), other.dataPoints.end());
    }
    else if (mode_ == Mode::MIN)
    {
        // MIN mode: merge by taking minimum times for each size
        for (const auto& pair : other.minTimesPerSize)
        {
            double size = pair.first;
            double time = pair.second;

            auto it = minTimesPerSize.find(size);
            if (it == minTimesPerSize.end())
            {
                // New size, add it
                minTimesPerSize[size] = time;
                sumX += size;
                sumY += time;
                sumXY += size * time;
                sumX2 += size * size;
                n++;
            }
            else if (time < it->second)
            {
                // Update to new minimum
                double oldTime = it->second;
                it->second     = time;

                // Update sums
                sumY  = sumY - oldTime + time;
                sumXY = sumXY - size * oldTime + size * time;
            }
            // If time >= current min, do nothing
        }
    }
}

/**
 * @brief Reset the accumulator to an empty state.
 */
void LinearRegression::clear()
{
    dataPoints.clear();
    minTimesPerSize.clear();
    sumX  = 0.0;
    sumY  = 0.0;
    sumXY = 0.0;
    sumX2 = 0.0;
    n     = 0;
}

/**
 * @brief Compute the least-squares slope and intercept.
 *
 * @param[out] slope Transfer-time slope in microseconds per byte.
 * @param[out] intercept Estimated fixed latency in microseconds.
 *
 * @return true when at least two non-degenerate points are available.
 */
bool LinearRegression::calculate(double& slope, double& intercept) const
{
    if (n < 2)
    {
        // Need at least two points to calculate a line
        slope     = 0.0;
        intercept = 0.0;
        return false;
    }

    double denominator = n * sumX2 - sumX * sumX;
    if (denominator == 0)
    {
        // Vertical line or all x-values are the same, cannot calculate slope
        slope     = 0.0;
        intercept = sumY / n;  // Average Y as intercept
        return false;
    }

    slope     = (n * sumXY - sumX * sumY) / denominator;
    intercept = (sumY * sumX2 - sumX * sumXY) / denominator;
    return true;
}

/**
 * @brief Check whether the accumulator has at least three unique sizes.
 *
 * @return true when enough size diversity exists for latency estimation.
 */
bool LinearRegression::hasAtLeastThreeDifferentSizes() const
{
    if (mode_ == Mode::AVG)
    {
        // Count unique X values in dataPoints
        std::set<double> uniqueSizes;
        for (const auto& point : dataPoints)
        {
            uniqueSizes.insert(point.first);
        }
        return uniqueSizes.size() >= 3;
    }
    else if (mode_ == Mode::MIN)
    {
        // Check number of entries in minTimesPerSize map
        return minTimesPerSize.size() >= 3;
    }
    return false;
}

/**
 * @brief Compute the R-squared goodness-of-fit value.
 *
 * @param[out] rSquared Coefficient of determination for the fitted line.
 *
 * @return true when the regression and fit statistics were computed.
 */
bool LinearRegression::calculateRSquared(double& rSquared) const
{
    if (n < 2)
    {
        rSquared = 0.0;
        return false;
    }

    double slope, intercept;
    if (!calculate(slope, intercept))
    {
        rSquared = 0.0;
        return false;
    }

    // Calculate total sum of squares (TSS) and residual sum of squares (RSS)
    double yMean = sumY / n;
    double tss   = 0.0;  // Total sum of squares
    double rss   = 0.0;  // Residual sum of squares

    // Use the appropriate data source based on mode
    if (mode_ == Mode::AVG)
    {
        for (const auto& point : dataPoints)
        {
            double x          = point.first;
            double y          = point.second;
            double yPredicted = intercept + slope * x;
            tss += (y - yMean) * (y - yMean);
            rss += (y - yPredicted) * (y - yPredicted);
        }
    }
    else if (mode_ == Mode::MIN)
    {
        for (const auto& pair : minTimesPerSize)
        {
            double x          = pair.first;
            double y          = pair.second;
            double yPredicted = intercept + slope * x;
            tss += (y - yMean) * (y - yMean);
            rss += (y - yPredicted) * (y - yPredicted);
        }
    }

    if (tss == 0.0)
    {
        // All y values are the same - perfect fit for constant function
        rSquared = 1.0;
        return true;
    }

    rSquared = 1.0 - (rss / tss);
    return true;
}
