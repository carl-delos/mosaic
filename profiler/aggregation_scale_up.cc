// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "aggregation.h"
#include "param.h"
#include "profiler_otel.h"
#include "scale_up_inference.h"

/**
 * @brief Finalize scale-up operations that completed without ProxyOp children.
 *
 * This path handles both collective and P2P operations that rely on inferred
 * transfer characteristics instead of proxy-thread transfer timing.
 *
 * @param[in] handleToOp Map of operation handles to in-progress operations.
 * @param[in] isColl True for collective operations, false for P2P operations.
 */
void WindowAggregator::finalizeScaleUpOperations(std::map<const void*, InProgressOperation>& handleToOp, bool isColl)
{
    double networkPct = (double)OTEL_GET_PARAM(ScaleUpNetworkPct);

    CommunicatorState* commState = nullptr;
    if (!handleToOp.empty())
    {
        const auto* eventPtr = static_cast<const otelEventHandle_t*>(handleToOp.begin()->first);
        commState            = const_cast<CommunicatorState*>(eventPtr ? eventPtr->commState : nullptr);
    }

    const bool isCudaGraphDriven = commState && commState->isScaleUpCudaGraphDriven();

    auto computeCollectiveTimeUs = [&](const void* opHandle, const InProgressOperation& op, bool& hasKernelEvents,
                                       const std::vector<otelEventHandle_t>*& kernelEvents) -> double
    {
        auto kernelIt   = kernelChByParent.find(opHandle);
        hasKernelEvents = (kernelIt != kernelChByParent.end() && !kernelIt->second.empty());
        kernelEvents    = hasKernelEvents ? &kernelIt->second : nullptr;

        double lastKernelEndTs = 0.0;
        if (hasKernelEvents)
        {
            for (const auto& kch : kernelIt->second)
                if (kch.endTs > lastKernelEndTs) lastKernelEndTs = kch.endTs;
        }
        return hasKernelEvents ? (lastKernelEndTs - op.startTs) : (op.endTs - op.startTs);
    };

    auto inferTransfers = [&](const InProgressOperation& op) -> InferredTransfers
    {
        if (isColl) return inferCollectiveTransfers(op.func, op.algo, op.bytes, op.nRanks, op.nChannels, networkPct);
        return inferP2PTransfers(op.bytes, op.nChannels, networkPct);
    };

    auto recordCollectiveCountTime = [&](const InProgressOperation& op, double collectiveTimeUs)
    {
        if (isColl)
            collectives[op.key].addCollective(op.bytes, collectiveTimeUs);
        else
            p2ps[op.key].addP2P(op.bytes, collectiveTimeUs);
    };

    auto recordTransferCacheBatch =
        [&](const InProgressOperation& op, int numTransfers, size_t perTransferBytes, double perTransferTimeUs)
    {
        const size_t totalBytes = (size_t)numTransfers * perTransferBytes;
        const double totalTime  = (double)numTransfers * perTransferTimeUs;
        if (isColl)
            collectives[op.key].addTransferBatch(numTransfers, totalBytes, totalTime);
        else
            p2ps[op.key].addTransferBatch(numTransfers, totalBytes, totalTime);
    };

    auto addRankChannelVolumeOnly =
        [&](const void* opHandle, const InProgressOperation& op, const InferredTransfers& inf)
    {
        if (inf.numTransfers <= 0 || inf.perTransferBytes == 0) return;

        const auto* eventPtr        = static_cast<const otelEventHandle_t*>(opHandle);
        const CommunicatorState* cs = eventPtr ? eventPtr->commState : nullptr;
        const int peer              = op.peer;

        const size_t totalBytes = (size_t)inf.numTransfers * inf.perTransferBytes;
        std::string rankKey     = getScaleUpRankTransferKey(cs, peer, !isColl);
        rankTransfers[rankKey].totalBytes += totalBytes;
        rankTransfers[rankKey].count += inf.numTransfers;

        int nCh  = inf.numChannels > 0 ? inf.numChannels : 1;
        int base = inf.numTransfers / nCh;
        int rem  = inf.numTransfers % nCh;
        for (int ch = 0; ch < nCh; ch++)
        {
            int transfersThisCh = base + (ch < rem ? 1 : 0);
            if (transfersThisCh <= 0) continue;
            std::string channelKey = getScaleUpChannelTransferKey(cs, peer, (uint8_t)ch, !isColl);
            channelTransfers[channelKey].totalBytes += (size_t)transfersThisCh * inf.perTransferBytes;
            channelTransfers[channelKey].count += transfersThisCh;
        }
    };

    for (auto& pair : handleToOp)
    {
        const void* opHandle    = pair.first;
        InProgressOperation& op = pair.second;

        if (!isColl && alltoAllP2PHandles.count(opHandle)) continue;

        if (op.seenProxyOps > 0) continue;

        bool hasKernelEvents                               = false;
        const std::vector<otelEventHandle_t>* kernelEvents = nullptr;
        double collectiveTimeUs = computeCollectiveTimeUs(opHandle, op, hasKernelEvents, kernelEvents);

        if (collectiveTimeUs <= 0)
        {
            OTEL_WARN(NCCL_INIT, "Skipping scale-up %s with invalid duration=%.2f us: %s, bytes=%zu",
                      isColl ? "Coll" : "P2P", collectiveTimeUs, op.key.c_str(), op.bytes);
            continue;
        }

        recordCollectiveCountTime(op, collectiveTimeUs);

        InferredTransfers inferred = inferTransfers(op);
        if (inferred.numTransfers <= 0 || inferred.perTransferBytes == 0)
        {
            OTEL_TRACE(NCCL_INIT, "Scale-up %s (no inferred transfers): %s, bytes=%zu, duration=%.2f us",
                       isColl ? "Coll" : "P2P", op.key.c_str(), op.bytes, collectiveTimeUs);
            continue;
        }

        if (isCudaGraphDriven)
        {
            double perTransferTime = 0.0;
            if (inferred.networkTimeFraction > 0.0 && inferred.numTransfers > 0)
            {
                const double networkTime = collectiveTimeUs * inferred.networkTimeFraction;
                perTransferTime          = networkTime / inferred.numTransfers;
            }

            recordTransferCacheBatch(op, inferred.numTransfers, inferred.perTransferBytes, perTransferTime);
            addRankChannelVolumeOnly(opHandle, op, inferred);

            OTEL_TRACE(NCCL_INIT,
                       "Finalized scale-up %s (CUDA Graph): %s, bytes=%zu, collectiveTime=%.2f us, transfers=%d, "
                       "perTransferBytes=%zu",
                       isColl ? "Coll" : "P2P", op.key.c_str(), op.bytes, collectiveTimeUs, inferred.numTransfers,
                       inferred.perTransferBytes);
            continue;
        }

        const double networkTime = collectiveTimeUs * inferred.networkTimeFraction;
        const double perTransferTime =
            (inferred.numTransfers > 0 && networkTime > 0.0) ? (networkTime / inferred.numTransfers) : 0.0;

        recordTransferCacheBatch(op, inferred.numTransfers, inferred.perTransferBytes, perTransferTime);

        const auto* eventPtr               = static_cast<const otelEventHandle_t*>(opHandle);
        const CommunicatorState* eventComm = eventPtr ? eventPtr->commState : nullptr;
        int peer                           = op.peer;
        std::string rankKey                = getScaleUpRankTransferKey(eventComm, peer, !isColl);

        if (kernelEvents && !kernelEvents->empty())
        {
            int nCh  = inferred.numChannels > 0 ? inferred.numChannels : 1;
            int base = inferred.numTransfers / nCh;
            int rem  = inferred.numTransfers % nCh;

            for (int ch = 0; ch < nCh; ch++)
            {
                int transfersThisCh = base + (ch < rem ? 1 : 0);
                if (transfersThisCh <= 0) continue;

                const otelEventHandle_t* matchingKernel = nullptr;
                for (const auto& kch : *kernelEvents)
                {
                    if ((int)kch.kernelCh.channelId == ch)
                    {
                        matchingKernel = &kch;
                        break;
                    }
                }
                if (!matchingKernel) continue;

                double channelStartTs = matchingKernel->startTs;
                double channelEndTs   = matchingKernel->endTs;
                double channelSpan    = channelEndTs - channelStartTs;
                if (channelSpan <= 0.0) continue;

                for (int i = 0; i < transfersThisCh; i++)
                {
                    double intervalStart = channelStartTs + (channelSpan * i) / transfersThisCh;
                    double intervalEnd   = intervalStart + perTransferTime;
                    if (intervalEnd > channelEndTs) intervalEnd = channelEndTs;

                    rankTransfers[rankKey].addTransferWithTimestamps(inferred.perTransferBytes, perTransferTime,
                                                                     intervalStart, intervalEnd);
                }
            }

            for (const auto& kch : *kernelEvents)
            {
                std::string channelKey = getScaleUpChannelTransferKey(eventComm, peer, kch.kernelCh.channelId, !isColl);
                double channelStartTs  = kch.startTs;
                double channelEndTs    = kch.endTs;

                int transfersThisCh = 0;
                if (inferred.numChannels > 0)
                {
                    int base        = inferred.numTransfers / inferred.numChannels;
                    int rem         = inferred.numTransfers % inferred.numChannels;
                    transfersThisCh = base + (((int)kch.kernelCh.channelId < rem) ? 1 : 0);
                }
                if (transfersThisCh <= 0) continue;

                double channelSpan = channelEndTs - channelStartTs;
                if (channelSpan <= 0.0) continue;

                for (int i = 0; i < transfersThisCh; i++)
                {
                    double intervalStart = channelStartTs + (channelSpan * i) / transfersThisCh;
                    double intervalEnd   = intervalStart + perTransferTime;
                    if (intervalEnd > channelEndTs) intervalEnd = channelEndTs;

                    channelTransfers[channelKey].addTransferWithTimestamps(inferred.perTransferBytes, perTransferTime,
                                                                           intervalStart, intervalEnd);
                }
            }
        }
        else
        {
            int nCh  = inferred.numChannels > 0 ? inferred.numChannels : 1;
            int base = inferred.numTransfers / nCh;
            int rem  = inferred.numTransfers % nCh;

            for (int ch = 0; ch < nCh; ch++)
            {
                int transfersThisCh = base + (ch < rem ? 1 : 0);
                if (transfersThisCh <= 0) continue;

                std::string channelKey = getScaleUpChannelTransferKey(eventComm, peer, (uint8_t)ch, !isColl);
                double channelStartTs  = op.startTs;
                double channelEndTs    = op.startTs + networkTime;
                double channelSpan     = channelEndTs - channelStartTs;
                if (channelSpan <= 0.0) continue;

                for (int i = 0; i < transfersThisCh; i++)
                {
                    double intervalStart = channelStartTs + (channelSpan * i) / transfersThisCh;
                    double intervalEnd   = intervalStart + perTransferTime;
                    if (intervalEnd > channelEndTs) intervalEnd = channelEndTs;

                    rankTransfers[rankKey].addTransferWithTimestamps(inferred.perTransferBytes, perTransferTime,
                                                                     intervalStart, intervalEnd);
                    channelTransfers[channelKey].addTransferWithTimestamps(inferred.perTransferBytes, perTransferTime,
                                                                           intervalStart, intervalEnd);
                }
            }
        }

        OTEL_TRACE(NCCL_INIT,
                   "Finalized scale-up %s: %s, bytes=%zu, collectiveTime=%.2f us, networkTime=%.2f us, transfers=%d, "
                   "perTransferBytes=%zu, perTransferTime=%.2f us, mode=%s",
                   isColl ? "Coll" : "P2P", op.key.c_str(), op.bytes, collectiveTimeUs, networkTime,
                   inferred.numTransfers, inferred.perTransferBytes, perTransferTime,
                   eventComm ? eventComm->getScaleUpExecModeString() : "unknown");
    }
}