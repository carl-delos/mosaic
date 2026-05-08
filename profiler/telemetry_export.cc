// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "telemetry_internal.h"

#ifdef ENABLE_OTEL

#include <opentelemetry/context/context.h>
#include <opentelemetry/metrics/sync_instruments.h>

#include <map>
#include <string>

#include "profiler_otel.h"

/**
 * @brief Export collective operation metrics to OpenTelemetry.
 *
 * @param[in] key Aggregation key for the collective operation.
 * @param[in] emit Values to record.
 * @param[in] eligibility Export guards derived from aggregation state.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] local_rank Local rank within the node.
 * @param[in] comm_hash Communicator hash for labeling.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] scale_up_exec_mode Scale-up execution mode string.
 * @param[in] export_tag Trace-only export tag.
 */
void exportCollectiveMetrics(const std::string& key, const CollectiveEmitView& emit,
                             const CollectiveExportEligibility& eligibility, int rank, const std::string& hostname,
                             int local_rank, uint64_t comm_hash, const std::string& gpu_pci_bus_id,
                             const std::string& gpu_uuid, const std::string& comm_type, int nranks,
                             const std::string& scale_up_exec_mode, [[maybe_unused]] const char* export_tag)
{
    (void)comm_type;
    std::string rank_str       = std::to_string(rank);
    std::string local_rank_str = std::to_string(local_rank);
    std::string communicator   = std::to_string(comm_hash);

    if (eligibility.export_core)
    {
        OTEL_TRACE(NCCL_INIT,
                   "Exporting Collective (%s): %s, count=%.0f, totalBytes=%.0f, totalTime=%.2f us -> AvgBytes=%.2f, "
                   "AvgTime=%.2f us",
                   export_tag, key.c_str(), emit.count, emit.totalBytes, emit.totalTimeUs, emit.avgBytes,
                   emit.avgTimeUs);

        std::map<std::string, std::string> labels = {
            {"communicator",       communicator          },
            {"operation",          key                   },
            {"rank",               rank_str              },
            {"hostname",           hostname              },
            {"local_rank",         local_rank_str        },
            {"gpu_pci_bus_id",     gpu_pci_bus_id        },
            {"gpu_uuid",           gpu_uuid              },
            {"comm_type",          "COLLECTIVE"          },
            {"comm_nranks",        std::to_string(nranks)},
            {"scale_up_exec_mode", scale_up_exec_mode    }
        };

        g_collBytesCounter->Add(emit.totalBytes, labels, opentelemetry::context::Context{});
        g_collTimeHist->Record(emit.avgTimeUs, labels, opentelemetry::context::Context{});
        g_collCountHist->Record((double)emit.count, labels, opentelemetry::context::Context{});

        if (eligibility.export_transfers)
        {
            g_collNumTransfersHist->Record(emit.avgNumTransfers, labels, opentelemetry::context::Context{});
            g_collTransferSizeHist->Record(emit.avgTransferSize, labels, opentelemetry::context::Context{});
            if (eligibility.export_transfer_time)
            {
                g_collTransferTimeHist->Record(emit.avgTransferTime, labels, opentelemetry::context::Context{});
            }

            OTEL_TRACE(NCCL_INIT,
                       "Exported Collective (%s): %s, AvgBytes: %.2f, AvgTime: %.2f us, "
                       "AvgNumTransfers: %.2f, AvgTransferSize: %.2f, AvgTransferTime: %.2f us",
                       export_tag, key.c_str(), emit.avgBytes, emit.avgTimeUs, emit.avgNumTransfers,
                       emit.avgTransferSize, emit.avgTransferTime);
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Exported Collective (%s): %s, AvgBytes: %.2f, AvgTime: %.2f us (no transfers)",
                       export_tag, key.c_str(), emit.avgBytes, emit.avgTimeUs);
        }
    }
}

/**
 * @brief Export P2P operation metrics to OpenTelemetry.
 *
 * @param[in] key Aggregation key for the P2P operation.
 * @param[in] emit Values to record.
 * @param[in] eligibility Export guards derived from aggregation state.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] local_rank Local rank within the node.
 * @param[in] comm_hash Communicator hash for labeling.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] scale_up_exec_mode Scale-up execution mode string.
 * @param[in] export_tag Trace-only export tag.
 */
void exportP2PMetrics(const std::string& key, const P2PEmitView& emit, const P2PExportEligibility& eligibility,
                      int rank, const std::string& hostname, int local_rank, uint64_t comm_hash,
                      const std::string& gpu_pci_bus_id, const std::string& gpu_uuid, const std::string& comm_type,
                      int nranks, const std::string& scale_up_exec_mode, [[maybe_unused]] const char* export_tag)
{
    (void)comm_type;
    std::string rank_str       = std::to_string(rank);
    std::string local_rank_str = std::to_string(local_rank);
    std::string communicator   = std::to_string(comm_hash);

    std::string src_pipeline = "";
    std::string dst_pipeline = "";
    size_t pipeline_pos      = key.find("_Pipeline");
    if (pipeline_pos != std::string::npos)
    {
        size_t src_start = pipeline_pos + 9;
        size_t to_pos    = key.find("ToPipeline", src_start);
        if (to_pos != std::string::npos)
        {
            src_pipeline     = key.substr(src_start, to_pos - src_start);
            size_t dst_start = to_pos + 10;
            size_t dst_end   = key.find("_", dst_start);
            dst_pipeline =
                (dst_end != std::string::npos) ? key.substr(dst_start, dst_end - dst_start) : key.substr(dst_start);
        }
    }

    std::string operation = "Pipeline" + src_pipeline + " -> Pipeline" + dst_pipeline;

    if (eligibility.export_core)
    {
        std::map<std::string, std::string> labels = {
            {"communicator",       communicator          },
            {"operation",          operation             },
            {"rank",               rank_str              },
            {"hostname",           hostname              },
            {"local_rank",         local_rank_str        },
            {"gpu_pci_bus_id",     gpu_pci_bus_id        },
            {"gpu_uuid",           gpu_uuid              },
            {"comm_type",          "P2P"                 },
            {"comm_nranks",        std::to_string(nranks)},
            {"scale_up_exec_mode", scale_up_exec_mode    }
        };

        g_p2pBytesHist->Record(emit.avgBytes, labels, opentelemetry::context::Context{});
        g_p2pTimeHist->Record(emit.avgTimeUs, labels, opentelemetry::context::Context{});

        if (eligibility.export_transfers)
        {
            g_p2pNumTransfersHist->Record(emit.avgNumTransfers, labels, opentelemetry::context::Context{});
            g_p2pTransferSizeHist->Record(emit.avgTransferSize, labels, opentelemetry::context::Context{});
            if (eligibility.export_transfer_time)
            {
                g_p2pTransferTimeHist->Record(emit.avgTransferTime, labels, opentelemetry::context::Context{});
            }

            OTEL_TRACE(NCCL_INIT,
                       "Exported P2P (%s): %s, AvgBytes: %.2f, AvgTime: %.2f us, "
                       "AvgNumTransfers: %.2f, AvgTransferSize: %.2f, AvgTransferTime: %.2f us",
                       export_tag, key.c_str(), emit.avgBytes, emit.avgTimeUs, emit.avgNumTransfers,
                       emit.avgTransferSize, emit.avgTransferTime);
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Exported P2P (%s): %s, AvgBytes: %.2f, AvgTime: %.2f us (no transfers)", export_tag,
                       key.c_str(), emit.avgBytes, emit.avgTimeUs);
        }
    }
}

/**
 * @brief Export rank transfer metrics to OpenTelemetry.
 *
 * @param[in] key Aggregation key for the rank transfer.
 * @param[in] emit Values to record.
 * @param[in] eligibility Export guards derived from aggregation state.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] local_rank Local rank within the node.
 * @param[in] scale_up_exec_mode Scale-up execution mode string.
 * @param[in] export_tag Trace-only export tag.
 */
void exportRankMetrics(const std::string& key, const RankEmitView& emit, const RankExportEligibility& eligibility,
                       int rank, const std::string& hostname, const std::string& gpu_pci_bus_id,
                       const std::string& gpu_uuid, const std::string& comm_type, int nranks, int local_rank,
                       const std::string& scale_up_exec_mode, [[maybe_unused]] const char* export_tag)
{
    (void)rank;
    std::string communicator = "";
    std::string source_rank  = "";
    std::string dest_rank    = "";

    size_t comm_pos     = key.find("Comm");
    size_t first_sep    = key.find("_", comm_pos + 4);
    size_t pipeline_pos = key.find("_Pipeline");
    size_t peer_pos     = key.find("_ToPeer");

    if (comm_pos != std::string::npos && first_sep != std::string::npos)
    {
        communicator = key.substr(comm_pos + 4, first_sep - comm_pos - 4);
    }

    if (pipeline_pos != std::string::npos && peer_pos == std::string::npos)
    {
        size_t src_start = pipeline_pos + 9;
        size_t to_pos    = key.find("_ToPipeline", src_start);
        if (to_pos != std::string::npos)
        {
            std::string src_pipeline = key.substr(src_start, to_pos - src_start);
            std::string dst_pipeline = key.substr(to_pos + 11);
            source_rank              = "Pipeline" + src_pipeline;
            dest_rank                = "Pipeline" + dst_pipeline;
        }
    }
    else if (peer_pos != std::string::npos)
    {
        size_t rank_pos = key.find("_Rank");
        if (rank_pos != std::string::npos)
        {
            source_rank = key.substr(rank_pos + 5, peer_pos - rank_pos - 5);
        }
        dest_rank = key.substr(peer_pos + 7);
    }

    std::string metricCommType = comm_type;
    if (pipeline_pos != std::string::npos && peer_pos == std::string::npos)
        metricCommType = "P2P";
    else if (peer_pos != std::string::npos)
        metricCommType = "COLLECTIVE";

    std::map<std::string, std::string> labels = {
        {"communicator",       communicator              },
        {"source_rank",        source_rank               },
        {"dest_rank",          dest_rank                 },
        {"hostname",           hostname                  },
        {"gpu_pci_bus_id",     gpu_pci_bus_id            },
        {"gpu_uuid",           gpu_uuid                  },
        {"comm_type",          metricCommType            },
        {"comm_nranks",        std::to_string(nranks)    },
        {"local_rank",         std::to_string(local_rank)},
        {"scale_up_exec_mode", scale_up_exec_mode        }
    };

    g_rankBytesCounter->Add(emit.totalBytes, labels, opentelemetry::context::Context{});

    if (eligibility.export_latency)
    {
        g_rankLatencyHist->Record(emit.latencyUs, labels, opentelemetry::context::Context{});
        OTEL_TRACE(NCCL_INIT, "Exported Rank Latency (%s): %s, Latency: %.2f us", export_tag, key.c_str(),
                   emit.latencyUs);
    }

    if (eligibility.export_rate)
    {
        g_rankRateHist->Record(emit.rateMBps, labels, opentelemetry::context::Context{});
        OTEL_TRACE(NCCL_INIT, "Exported Rank Rate (%s): %s, Bytes: %llu, ActiveTime: %.2f us, Rate: %.2f MB/s",
                   export_tag, key.c_str(), static_cast<unsigned long long>(emit.totalBytes), emit.activeTimeUs,
                   emit.rateMBps);
    }
    else
    {
        OTEL_TRACE(NCCL_INIT, "Exported Rank Metrics (%s): %s, Bytes: %llu (no rate data)", export_tag, key.c_str(),
                   static_cast<unsigned long long>(emit.totalBytes));
    }
}

/**
 * @brief Export per-channel transfer metrics to OpenTelemetry.
 *
 * @param[in] key Aggregation key for the channel transfer.
 * @param[in] emit Values to record.
 * @param[in] eligibility Export guards derived from aggregation state.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] local_rank Local rank within the node.
 * @param[in] scale_up_exec_mode Scale-up execution mode string.
 * @param[in] export_tag Trace-only export tag.
 */
void exportTransferMetrics(const std::string& key, const TransferEmitView& emit,
                           const TransferExportEligibility& eligibility, int rank, const std::string& hostname,
                           const std::string& gpu_pci_bus_id, const std::string& gpu_uuid, const std::string& comm_type,
                           int nranks, int local_rank, const std::string& scale_up_exec_mode,
                           [[maybe_unused]] const char* export_tag)
{
    (void)rank;
    std::string communicator = "";
    std::string source_rank  = "";
    std::string dest_rank    = "";
    std::string channel      = "";

    size_t comm_pos     = key.find("Comm");
    size_t first_sep    = key.find("_", comm_pos + 4);
    size_t pipeline_pos = key.find("_Pipeline");
    size_t peer_pos     = key.find("_ToPeer");
    size_t chnl_pos     = key.find("_Chnl");

    if (comm_pos != std::string::npos && first_sep != std::string::npos)
    {
        communicator = key.substr(comm_pos + 4, first_sep - comm_pos - 4);
    }

    if (pipeline_pos != std::string::npos && peer_pos == std::string::npos)
    {
        size_t src_start = pipeline_pos + 9;
        size_t to_pos    = key.find("_ToPipeline", src_start);
        if (to_pos != std::string::npos)
        {
            std::string src_pipeline = key.substr(src_start, to_pos - src_start);
            std::string dst_pipeline = (chnl_pos != std::string::npos) ? key.substr(to_pos + 11, chnl_pos - to_pos - 11)
                                                                       : key.substr(to_pos + 11);
            source_rank              = "Pipeline" + src_pipeline;
            dest_rank                = "Pipeline" + dst_pipeline;
        }
    }
    else if (peer_pos != std::string::npos)
    {
        size_t rank_pos = key.find("_Rank");
        if (rank_pos != std::string::npos)
        {
            source_rank = key.substr(rank_pos + 5, peer_pos - rank_pos - 5);
        }
        if (chnl_pos != std::string::npos)
        {
            dest_rank = key.substr(peer_pos + 7, chnl_pos - peer_pos - 7);
        }
    }

    std::string metricCommType = comm_type;
    if (pipeline_pos != std::string::npos && peer_pos == std::string::npos)
        metricCommType = "P2P";
    else if (peer_pos != std::string::npos)
        metricCommType = "COLLECTIVE";

    if (chnl_pos != std::string::npos)
    {
        channel = key.substr(chnl_pos + 5);
    }

    if (eligibility.export_channel_metrics)
    {
        std::map<std::string, std::string> labels = {
            {"communicator",       communicator              },
            {"source_rank",        source_rank               },
            {"dest_rank",          dest_rank                 },
            {"channel",            channel                   },
            {"hostname",           hostname                  },
            {"gpu_pci_bus_id",     gpu_pci_bus_id            },
            {"gpu_uuid",           gpu_uuid                  },
            {"comm_type",          metricCommType            },
            {"comm_nranks",        std::to_string(nranks)    },
            {"local_rank",         std::to_string(local_rank)},
            {"scale_up_exec_mode", scale_up_exec_mode        }
        };

        g_transferSizeHist->Record(emit.avgSize, labels, opentelemetry::context::Context{});
        if (eligibility.export_avg_time)
        {
            g_transferTimeHist->Record(emit.avgTime, labels, opentelemetry::context::Context{});
        }

        if (eligibility.export_latency)
        {
            g_transferLatencyHist->Record(emit.latencyUs, labels, opentelemetry::context::Context{});
            OTEL_TRACE(NCCL_INIT, "Exported Transfer (%s): %s, AvgSize: %.2f, AvgTime: %.2f us, Latency: %.2f us",
                       export_tag, key.c_str(), emit.avgSize, emit.avgTime, emit.latencyUs);
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Exported Transfer (%s): %s, AvgSize: %.2f, AvgTime: %.2f us", export_tag,
                       key.c_str(), emit.avgSize, emit.avgTime);
        }
    }
}

#endif  // ENABLE_OTEL