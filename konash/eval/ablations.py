from __future__ import annotations


class SearchEnvironmentAblation:
    component = None
    setting = None
    benchmark_score = None
    benchmark_recall = None

    def __init__(self, component=None, setting=None, benchmark_score=None, benchmark_recall=None):
        self.component = component
        self.setting = setting
        self.benchmark_score = benchmark_score
        self.benchmark_recall = benchmark_recall


class CompressionTransferResult:
    search_model = None
    compression_model = None
    browsecomp_score = None

    def __init__(self, search_model=None, compression_model=None, browsecomp_score=None):
        self.search_model = search_model
        self.compression_model = compression_model
        self.browsecomp_score = browsecomp_score


class SharpeningAnalysis:
    max_at_k_curve = None
    pass_rate_flow = None
    transition_matrix = None
    supports_partial_solved_unsolved = True


class SearchScalingSweep:
    search_horizons = None
    retrieval_counts = None
    scores_by_horizon = None
    scores_by_retrieval_count = None


class AblationRegistry:
    search_environment = {
        ("compression", "with"): SearchEnvironmentAblation(
            component="compression",
            setting="with",
            benchmark_score=0.570,
            benchmark_recall=0.681,
        ),
        ("compression", "without"): SearchEnvironmentAblation(
            component="compression",
            setting="without",
            benchmark_score=0.389,
            benchmark_recall=0.503,
        ),
        ("retrieval", "Qwen3-Embedding-8B"): SearchEnvironmentAblation(
            component="retrieval",
            setting="Qwen3-Embedding-8B",
            benchmark_score=0.570,
            benchmark_recall=0.681,
        ),
        ("retrieval", "Vector Search (GTE-large + hybrid)"): SearchEnvironmentAblation(
            component="retrieval",
            setting="Vector Search (GTE-large + hybrid)",
            benchmark_score=0.568,
            benchmark_recall=0.698,
        ),
    }

    compression_transfer = {
        ("GLM 4.5 Air", "GLM 4.5 Air"): CompressionTransferResult(
            search_model="GLM 4.5 Air",
            compression_model="GLM 4.5 Air",
            browsecomp_score=0.44,
        ),
        ("GLM 4.5 Air", "KARL-BCP"): CompressionTransferResult(
            search_model="GLM 4.5 Air",
            compression_model="KARL-BCP",
            browsecomp_score=0.54,
        ),
        ("KARL-BCP", "GLM 4.5 Air"): CompressionTransferResult(
            search_model="KARL-BCP",
            compression_model="GLM 4.5 Air",
            browsecomp_score=0.46,
        ),
        ("KARL-BCP", "KARL-BCP"): CompressionTransferResult(
            search_model="KARL-BCP",
            compression_model="KARL-BCP",
            browsecomp_score=0.57,
        ),
    }
