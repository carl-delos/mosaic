// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "scale_up_inference.h"

#include <cmath>
#include <cstring>

/**
 * @brief Infer transfer characteristics for a collective operation on scale-up.
 *
 * Uses the collective type, data size, rank count and channel count to estimate:
 * - Per-transfer size
 * - Total number of transfers for this rank
 * - Total bytes transferred by this rank on the internal network
 *
 * @param[in] func Collective function name.
 * @param[in] algo Algorithm name.
 * @param[in] collectiveBytes Data size from the profiler event.
 * @param[in] nRanks Number of ranks in the communicator.
 * @param[in] nChannels Number of channels used by this collective.
 * @param[in] networkPct Percentage of collective time assumed spent on networking.
 *
 * @return Inferred transfer parameters for the collective.
 */
InferredTransfers inferCollectiveTransfers(const char* func, const char* algo, size_t collectiveBytes, int nRanks,
                                           uint8_t nChannels, double networkPct)
{
    InferredTransfers result   = {};
    result.networkTimeFraction = (networkPct > 0 && networkPct <= 100) ? networkPct / 100.0 : 1.0;
    result.numChannels         = nChannels > 0 ? nChannels : 1;

    if (collectiveBytes == 0 || nRanks <= 1)
    {
        result.perTransferBytes = 0;
        result.numTransfers     = 0;
        result.totalRankBytes   = 0;
        result.stepsPerRank     = 0;
        return result;
    }

    size_t nBytesGlobal      = collectiveBytes;
    double trafficMultiplier = 1.0;
    int stepsPerRank         = 1;

    if (func)
    {
        if (strstr(func, "AllReduce"))
        {
            nBytesGlobal      = collectiveBytes;
            trafficMultiplier = 2.0;
            stepsPerRank      = 2 * (nRanks - 1);
        }
        else if (strstr(func, "AllGather"))
        {
            nBytesGlobal      = collectiveBytes * (size_t)nRanks;
            trafficMultiplier = (double)nRanks;
            stepsPerRank      = nRanks - 1;
        }
        else if (strstr(func, "ReduceScatter"))
        {
            nBytesGlobal      = collectiveBytes * (size_t)nRanks;
            trafficMultiplier = (double)nRanks;
            stepsPerRank      = nRanks - 1;
        }
        else if (strstr(func, "Broadcast") || strstr(func, "Reduce"))
        {
            nBytesGlobal      = collectiveBytes;
            trafficMultiplier = 1.0;
            stepsPerRank      = 1;
        }
    }

    result.stepsPerRank = stepsPerRank;
    result.totalRankBytes =
        (size_t)((double)collectiveBytes * trafficMultiplier * (double)(nRanks - 1) / (double)nRanks);

    bool isTree = algo && (strstr(algo, "TREE") || strstr(algo, "Tree"));

    if (isTree)
    {
        size_t treePerChannel = nBytesGlobal / (size_t)result.numChannels;
        if (treePerChannel == 0) treePerChannel = 1;

        size_t treeTransferSize =
            treePerChannel <= SCALE_UP_TREE_TRANSFER_BYTES ? treePerChannel : SCALE_UP_TREE_TRANSFER_BYTES;

        result.perTransferBytes = treeTransferSize;
        result.numTransfers =
            result.totalRankBytes > 0 ? (int)std::ceil((double)result.totalRankBytes / treeTransferSize) : 0;
        return result;
    }

    size_t baseTransferSize = nBytesGlobal / (size_t)nRanks / (size_t)result.numChannels;
    if (baseTransferSize == 0) baseTransferSize = 1;

    int slicesPerChunk = 1;
    if (baseTransferSize >= SCALE_UP_SLICE_SPLIT_THRESHOLD)
    {
        slicesPerChunk = 2;
        baseTransferSize /= 2;
    }

    int numSubTransfers = 1;
    size_t perTransfer  = baseTransferSize;
    if (baseTransferSize > SCALE_UP_MAX_TRANSFER_BYTES)
    {
        numSubTransfers = (int)std::ceil((double)baseTransferSize / SCALE_UP_MAX_TRANSFER_BYTES);
        perTransfer     = SCALE_UP_MAX_TRANSFER_BYTES;
    }

    result.perTransferBytes = perTransfer;
    result.numTransfers     = stepsPerRank * result.numChannels * slicesPerChunk * numSubTransfers;
    return result;
}

/**
 * @brief Infer transfer characteristics for a P2P operation on scale-up.
 *
 * @param[in] p2pBytes Total bytes in the P2P operation.
 * @param[in] nChannels Number of channels used.
 * @param[in] networkPct Percentage of P2P time assumed spent on networking.
 *
 * @return Inferred transfer parameters for the P2P operation.
 */
InferredTransfers inferP2PTransfers(size_t p2pBytes, uint8_t nChannels, double networkPct)
{
    InferredTransfers result   = {};
    result.networkTimeFraction = (networkPct > 0 && networkPct <= 100) ? networkPct / 100.0 : 1.0;
    result.numChannels         = nChannels > 0 ? nChannels : 1;
    result.stepsPerRank        = 1;

    if (p2pBytes == 0)
    {
        result.perTransferBytes = 0;
        result.numTransfers     = 0;
        result.totalRankBytes   = 0;
        return result;
    }

    size_t perChannelBytes = p2pBytes / (size_t)result.numChannels;
    if (perChannelBytes == 0) perChannelBytes = 1;

    int slicesPerChunk = 1;
    if (perChannelBytes >= SCALE_UP_SLICE_SPLIT_THRESHOLD)
    {
        slicesPerChunk = 2;
        perChannelBytes /= 2;
    }

    int numSubTransfers = 1;
    size_t perTransfer  = perChannelBytes;
    if (perChannelBytes > SCALE_UP_MAX_TRANSFER_BYTES)
    {
        numSubTransfers = (int)std::ceil((double)perChannelBytes / SCALE_UP_MAX_TRANSFER_BYTES);
        perTransfer     = SCALE_UP_MAX_TRANSFER_BYTES;
    }

    result.perTransferBytes = perTransfer;
    result.numTransfers     = result.numChannels * slicesPerChunk * numSubTransfers;
    result.totalRankBytes   = p2pBytes;
    return result;
}