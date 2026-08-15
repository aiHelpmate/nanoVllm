"""Benchmark comparison: nano-vllm vs HuggingFace Transformers.

Compares inference performance across various configurations:
- Different batch sizes (1, 2, 4, 8)
- Different output lengths (20, 50, 100 tokens)
- Various prompt types (short, medium, long)
- nano-vllm with PagedAttention vs legacy vs HuggingFace
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test prompts of varying lengths and topics
PROMPTS = {
    "short": [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Name a color.",
        "Say something funny.",
        "What is AI?",
        "Count to five.",
        "Define happiness.",
        "What is the sun?",
    ],
    "medium": [
        "Explain the concept of machine learning in simple terms.",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis step by step.",
        "What are the benefits of regular exercise for mental health?",
        "Explain how the internet works to a beginner.",
        "What is the significance of the Renaissance period?",
        "Describe the water cycle and its importance.",
        "What are the key principles of object-oriented programming?",
    ],
    "long": [
        "Write a detailed explanation of how neural networks learn, including the concepts of forward propagation, backpropagation, and gradient descent. Include examples where appropriate.",
        "Explain the history of artificial intelligence from its inception to modern day, covering key milestones, important researchers, and breakthrough technologies that shaped the field.",
        "Describe the complete process of software development lifecycle, from requirements gathering to deployment and maintenance, including best practices for each phase.",
        "Write a comprehensive guide on climate change, covering its causes, effects on ecosystems and human society, and potential solutions that individuals and governments can implement.",
        "Explain the fundamentals of quantum computing, how it differs from classical computing, its potential applications, and the current challenges in building practical quantum computers.",
        "Describe the architecture of modern large language models, including transformer architecture, attention mechanisms, and the training process used to create models like GPT.",
        "Write about the evolution of the internet from ARPANET to Web 3.0, covering major technological advances, social impacts, and future predictions for connectivity.",
        "Explain the principles of distributed systems, including concepts like consistency, availability, partition tolerance, and common patterns used in building scalable applications.",
    ],
    "extreme": [
        # ~500-800 tokens each - tests long context handling
        """You are a senior software architect reviewing a complex distributed system. The system consists of multiple microservices including: an API gateway handling authentication and rate limiting, a user service managing profiles and preferences, an order service processing transactions, an inventory service tracking stock levels, a notification service for emails and push notifications, and an analytics service collecting metrics. Each service communicates via both synchronous REST APIs and asynchronous message queues using RabbitMQ. The database layer uses PostgreSQL for transactional data, Redis for caching, and Elasticsearch for search functionality. The entire system is deployed on Kubernetes with auto-scaling policies based on CPU and memory utilization. Recently, the team has observed intermittent latency spikes during peak traffic hours, occasional message queue backlogs, and some database connection pool exhaustion errors. The monitoring stack includes Prometheus for metrics, Grafana for visualization, and Jaeger for distributed tracing. Given this architecture, please analyze the potential root causes of these performance issues and provide a comprehensive remediation plan that addresses immediate fixes, medium-term optimizations, and long-term architectural improvements. Consider factors such as connection pooling strategies, caching policies, queue consumer scaling, database query optimization, and potential service decomposition.""",

        """The field of artificial intelligence has undergone remarkable transformations since its inception in the mid-20th century. Beginning with the Dartmouth Conference in 1956, where the term "artificial intelligence" was first coined by John McCarthy, the field has experienced multiple cycles of optimism and disappointment, often referred to as "AI winters" and "AI summers." The early years focused on symbolic AI and expert systems, attempting to encode human knowledge into rule-based systems. Projects like MYCIN for medical diagnosis and DENDRAL for chemical analysis showed promising results but ultimately proved too brittle for real-world applications. The 1980s saw a resurgence with expert systems in business applications, followed by another winter when these systems failed to meet expectations. The breakthrough came with machine learning approaches, particularly neural networks, which had existed since the 1940s with McCulloch and Pitts' work but gained practical significance only with increased computational power and data availability. The ImageNet competition in 2012, where AlexNet dramatically outperformed traditional computer vision methods, marked the beginning of the deep learning revolution. This was followed by advances in natural language processing, culminating in the transformer architecture introduced in 2017 and subsequent models like BERT, GPT, and their successors. Today, large language models demonstrate emergent capabilities that surprise even their creators, raising fundamental questions about the nature of intelligence, consciousness, and the future relationship between humans and AI systems. Please continue this analysis by discussing the current state of AI research, major challenges, ethical considerations, and predictions for the next decade.""",

        """In the realm of modern web application development, security considerations span multiple layers and require constant vigilance against evolving threats. Starting from the network layer, applications must implement proper TLS configuration with modern cipher suites, certificate pinning for mobile clients, and protection against man-in-the-middle attacks. At the application layer, developers must guard against the OWASP Top 10 vulnerabilities including SQL injection, cross-site scripting (XSS), cross-site request forgery (CSRF), insecure direct object references, security misconfiguration, sensitive data exposure, missing function-level access control, and using components with known vulnerabilities. Authentication systems require careful implementation of password hashing using algorithms like bcrypt or Argon2, secure session management with proper cookie attributes (HttpOnly, Secure, SameSite), multi-factor authentication, and protection against brute force attacks through rate limiting and account lockout policies. Authorization must follow the principle of least privilege, implementing role-based or attribute-based access control consistently across all endpoints. API security demands attention to authentication tokens (JWT best practices, token rotation, short expiration times), input validation, output encoding, and proper error handling that doesn't leak sensitive information. Infrastructure security involves hardening servers, keeping dependencies updated, implementing web application firewalls, setting up intrusion detection systems, and maintaining comprehensive logging and monitoring. Given these requirements, design a comprehensive security architecture for a fintech application handling sensitive financial data, including specific technologies, configurations, and processes for maintaining security posture over time.""",

        """The global climate crisis represents one of the most complex challenges facing humanity, requiring coordinated action across scientific, political, economic, and social domains. Scientific consensus, documented through IPCC reports, establishes that anthropogenic greenhouse gas emissions have raised global average temperatures by approximately 1.1 degrees Celsius above pre-industrial levels, with projections suggesting we may exceed 1.5 degrees within the next two decades without dramatic intervention. The consequences are already observable: more frequent and intense extreme weather events including hurricanes, droughts, and wildfires; rising sea levels threatening coastal communities; disruption of ecosystems leading to biodiversity loss; and impacts on agricultural productivity affecting food security. The Paris Agreement established a framework for international cooperation, with countries submitting Nationally Determined Contributions outlining their emission reduction commitments, though current pledges fall short of limiting warming to 1.5 or even 2 degrees. Mitigation strategies encompass transitioning to renewable energy sources (solar, wind, hydroelectric, nuclear), improving energy efficiency across buildings, transportation, and industry, developing carbon capture and storage technologies, protecting and restoring forests and other carbon sinks, and transforming agricultural practices. Adaptation measures include building resilient infrastructure, developing drought-resistant crops, implementing early warning systems for extreme weather, and planning for managed retreat from vulnerable coastal areas. The economic implications involve stranded assets in fossil fuel industries, opportunities in green technologies, carbon pricing mechanisms, and ensuring a just transition for workers and communities dependent on carbon-intensive industries. Analyze the interplay between these factors and propose a comprehensive policy framework that balances environmental effectiveness, economic feasibility, and social equity.""",

        """Database design and optimization represent critical skills for building scalable applications. Consider a social media platform with the following requirements: users can create profiles, post content (text, images, videos), follow other users, like and comment on posts, receive notifications, and search for content and users. The platform must support millions of daily active users with sub-second response times for common operations. Relational database design must carefully consider normalization levels, choosing between third normal form for data integrity versus strategic denormalization for read performance. The schema must efficiently model the social graph (follower relationships), content hierarchy (posts, comments, replies), and user interactions (likes, shares). Indexing strategy requires analyzing query patterns to create appropriate B-tree indexes for equality and range queries, composite indexes for multi-column filters, and potentially partial indexes for filtered queries. For the social graph, consider whether to use adjacency lists, nested sets, or materialized path representations, each with different trade-offs for read versus write performance. The notification system requires careful design to avoid the thundering herd problem when popular users post content. Caching layers using Redis or Memcached must implement appropriate invalidation strategies, considering cache-aside, write-through, and write-behind patterns. For search functionality, integration with Elasticsearch provides full-text search capabilities but requires maintaining synchronization with the primary database. Sharding strategies must consider how to partition data (by user ID, geographic region, or time-based) while maintaining the ability to efficiently query across shards. Given these requirements and constraints, provide a detailed database architecture including schema design, indexing strategy, caching approach, and scaling plan.""",

        """The evolution of programming paradigms reflects our ongoing effort to manage software complexity and improve developer productivity. Imperative programming, the earliest paradigm, directly maps to machine execution with sequential statements modifying program state. Structured programming, advocated by Dijkstra, introduced control flow constructs (if-then-else, loops) as alternatives to goto statements, enabling clearer reasoning about program behavior. Object-oriented programming emerged from Simula and Smalltalk, organizing code around objects that encapsulate state and behavior, with inheritance and polymorphism enabling code reuse and abstraction. The SOLID principles (Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, Dependency Inversion) provide guidelines for creating maintainable object-oriented designs. Functional programming, rooted in lambda calculus, emphasizes immutability, pure functions without side effects, and higher-order functions that take or return other functions. Languages like Haskell enforce purity while others like Scala and F# offer hybrid approaches. The benefits include easier reasoning about code behavior, natural parallelization, and powerful abstractions like monads for handling effects. Reactive programming addresses asynchronous data streams, with frameworks like RxJS and Project Reactor providing operators for transforming and combining event sequences. Concurrent and parallel programming patterns have evolved from threads and locks to higher-level abstractions like actors (Erlang, Akka), communicating sequential processes (Go channels), and software transactional memory. Modern languages often combine paradigms: Python supports procedural, object-oriented, and functional styles; Rust combines systems programming with functional concepts and ownership-based memory safety. Analyze how these paradigms have influenced modern software architecture patterns including microservices, event sourcing, and serverless computing.""",

        """Quantum computing represents a fundamental shift in computational paradigms, leveraging quantum mechanical phenomena to process information in ways impossible for classical computers. Unlike classical bits that exist in definite states of 0 or 1, quantum bits (qubits) can exist in superpositions of both states simultaneously, described by complex probability amplitudes. When multiple qubits are entangled, measuring one instantaneously affects the others regardless of physical distance, a phenomenon Einstein called "spooky action at a distance." These properties enable quantum parallelism, where a quantum computer can evaluate multiple possibilities simultaneously. Quantum algorithms exploit this parallelism: Shor's algorithm can factor large integers exponentially faster than classical algorithms, threatening RSA encryption; Grover's algorithm provides quadratic speedup for unstructured search problems. The challenge lies in maintaining quantum coherence while performing computations. Qubits are extremely sensitive to environmental disturbances (decoherence), requiring sophisticated error correction codes that encode logical qubits across many physical qubits. Current quantum computers, including those from IBM, Google, and IonQ, use various physical implementations: superconducting circuits operating at millikelvin temperatures, trapped ions manipulated by lasers, photonic systems using light particles, and topological approaches promising inherent error protection. We are currently in the Noisy Intermediate-Scale Quantum (NISQ) era, with devices containing tens to hundreds of noisy qubits. Applications being explored include quantum simulation of molecular systems for drug discovery and materials science, optimization problems in logistics and finance, quantum machine learning, and cryptography. Post-quantum cryptography research develops classical algorithms resistant to quantum attacks. Discuss the current state of quantum computing hardware and software, near-term applications, and the path toward fault-tolerant quantum computation.""",

        """The human brain, containing approximately 86 billion neurons connected by trillions of synapses, represents the most complex known structure in the universe. Neuroscience has made remarkable progress in understanding brain function at multiple scales, from molecular mechanisms of synaptic transmission to systems-level organization of cognitive functions. At the cellular level, neurons communicate through electrochemical signals: action potentials propagate along axons through voltage-gated ion channels, triggering neurotransmitter release at synapses. Different neurotransmitter systems (glutamate, GABA, dopamine, serotonin, acetylcholine) modulate neural activity and underlie various cognitive and emotional processes. Synaptic plasticity, captured by Hebb's principle ("neurons that fire together wire together"), provides the substrate for learning and memory. Long-term potentiation (LTP) and long-term depression (LTD) involve complex molecular cascades affecting receptor trafficking and gene expression. Brain imaging technologies have revolutionized our understanding: functional MRI reveals blood oxygen level-dependent signals correlated with neural activity; electroencephalography captures millisecond-scale electrical dynamics; magnetoencephalography provides similar temporal resolution with better spatial localization. Connectomics efforts map the complete wiring diagrams of brains, from the 302-neuron C. elegans nervous system to ongoing human connectome projects. Computational neuroscience develops mathematical models at various levels of abstraction: detailed biophysical models of individual neurons, neural network models capturing circuit dynamics, and cognitive architectures attempting to explain high-level mental functions. The relationship between brain and mind remains deeply puzzling: how does subjective conscious experience arise from physical neural processes? This "hard problem of consciousness" continues to challenge philosophers and scientists alike. Discuss current theories of consciousness, the neural correlates of awareness, and implications for artificial intelligence and machine consciousness.""",
    ],
}


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    backend: str  # "hf", "nano_vllm_paged", "nano_vllm_legacy"
    batch_size: int
    max_tokens: int
    prompt_type: str
    num_prompts: int
    total_time_sec: float
    tokens_generated: int
    throughput_tokens_per_sec: float
    avg_latency_per_prompt_ms: float
    peak_memory_mb: Optional[float] = None


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    model_path: str
    batch_sizes: List[int]
    max_tokens_list: List[int]
    prompt_types: List[str]
    device: str
    dtype: torch.dtype
    warmup_runs: int = 2
    num_runs: int = 3
    multi_gpu: bool = False


def get_gpu_memory_mb(multi_gpu: bool = False) -> Optional[float]:
    """Get current GPU memory usage in MB.

    Args:
        multi_gpu: If True, sum memory across all GPUs.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if multi_gpu:
            total_memory = sum(
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            )
            return total_memory / (1024 * 1024)
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None


def reset_gpu_memory(multi_gpu: bool = False):
    """Reset GPU memory stats and run garbage collection.

    Args:
        multi_gpu: If True, reset stats on all GPUs.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if multi_gpu:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        else:
            torch.cuda.reset_peak_memory_stats()


def benchmark_huggingface(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int,
    device: str,
    multi_gpu: bool = False,
) -> tuple[float, int]:
    """Benchmark HuggingFace generate().

    Returns:
        Tuple of (total_time_seconds, total_tokens_generated)
    """
    # Tokenize all prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # For multi-GPU, move inputs to the model's input device
    if multi_gpu:
        inputs = inputs.to(model.device)
    else:
        inputs = inputs.to(device)

    input_len = inputs.input_ids.shape[1]

    # Generate
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy for fair comparison
            pad_token_id=tokenizer.eos_token_id,
        )
    # Synchronize all GPUs for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Count generated tokens (excluding input)
    total_generated = sum(
        len(output) - input_len for output in outputs
    )

    return elapsed, total_generated


def benchmark_nano_vllm(
    engine,
    prompts: List[str],
    max_tokens: int,
) -> tuple[float, int]:
    """Benchmark nano-vllm generate_batch().

    Returns:
        Tuple of (total_time_seconds, total_tokens_generated)
    """
    # Add all requests
    for prompt in prompts:
        engine.add_request(prompt, max_tokens)

    start = time.perf_counter()
    # Run to completion and get GenerationOutput objects with accurate token counts
    outputs = engine.run_to_completion()
    if engine.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Use the accurate generated_tokens count from the engine
    total_generated = sum(output.generated_tokens for output in outputs)

    return elapsed, total_generated


def unload_model(model, multi_gpu: bool = False):
    """Unload a model and free GPU memory.

    Args:
        model: The model to unload (can be HuggingFace model or nano-vllm engine)
        multi_gpu: Whether multi-GPU mode is enabled
    """
    if model is None:
        return

    # Delete the model
    del model

    # Run garbage collection
    gc.collect()

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if multi_gpu:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        else:
            torch.cuda.reset_peak_memory_stats()

    print("  Model unloaded and GPU memory cleared.")


def run_benchmark(
    config: BenchmarkConfig,
    skip_hf: bool = False,
    skip_nano_paged: bool = False,
    skip_nano_legacy: bool = False,
) -> List[BenchmarkResult]:
    """Run full benchmark suite.

    Each backend is loaded, benchmarked, then unloaded before the next backend
    to avoid memory issues from having multiple models loaded simultaneously.

    Args:
        config: Benchmark configuration
        skip_hf: Skip HuggingFace benchmarks
        skip_nano_paged: Skip nano-vllm with PagedAttention
        skip_nano_legacy: Skip nano-vllm legacy mode

    Returns:
        List of benchmark results
    """
    results = []

    # Build list of configurations to test
    test_configs = []
    for batch_size in config.batch_sizes:
        for max_tokens in config.max_tokens_list:
            for prompt_type in config.prompt_types:
                test_configs.append((batch_size, max_tokens, prompt_type))

    total_configs = len(test_configs)

    # ========== Benchmark HuggingFace ==========
    if not skip_hf:
        print("\n" + "=" * 60)
        print("BENCHMARKING: HuggingFace")
        print("=" * 60)

        print("Loading HuggingFace model...")
        hf_tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        if config.multi_gpu:
            print(f"  Using multi-GPU with device_map='auto' ({torch.cuda.device_count()} GPUs available)")
            hf_model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=config.dtype,
                device_map="auto",
            )
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=config.dtype,
            ).to(config.device)
        hf_model.eval()
        print("HuggingFace model loaded.")

        for idx, (batch_size, max_tokens, prompt_type) in enumerate(test_configs, 1):
            prompts = PROMPTS[prompt_type][:batch_size]
            print(f"\n[{idx}/{total_configs}] batch={batch_size}, "
                  f"max_tokens={max_tokens}, prompts={prompt_type}")

            reset_gpu_memory(config.multi_gpu)

            # Warmup
            for _ in range(config.warmup_runs):
                benchmark_huggingface(
                    hf_model, hf_tokenizer, prompts, max_tokens,
                    config.device, config.multi_gpu
                )

            # Timed runs
            times = []
            tokens = []
            for _ in range(config.num_runs):
                reset_gpu_memory(config.multi_gpu)
                t, tok = benchmark_huggingface(
                    hf_model, hf_tokenizer, prompts, max_tokens,
                    config.device, config.multi_gpu
                )
                times.append(t)
                tokens.append(tok)

            avg_time = sum(times) / len(times)
            avg_tokens = sum(tokens) / len(tokens)
            peak_mem = get_gpu_memory_mb(config.multi_gpu)

            results.append(BenchmarkResult(
                backend="huggingface",
                batch_size=batch_size,
                max_tokens=max_tokens,
                prompt_type=prompt_type,
                num_prompts=len(prompts),
                total_time_sec=avg_time,
                tokens_generated=int(avg_tokens),
                throughput_tokens_per_sec=avg_tokens / avg_time,
                avg_latency_per_prompt_ms=(avg_time / len(prompts)) * 1000,
                peak_memory_mb=peak_mem,
            ))
            print(f"  HuggingFace: {avg_tokens/avg_time:.1f} tok/s, {avg_time:.3f}s")

        # Unload HuggingFace model before loading next
        print("\nUnloading HuggingFace model...")
        unload_model(hf_model, config.multi_gpu)
        del hf_tokenizer
        gc.collect()

    # ========== Benchmark nano-vllm with PagedAttention ==========
    if not skip_nano_paged:
        print("\n" + "=" * 60)
        print("BENCHMARKING: nano-vllm (PagedAttention)")
        print("=" * 60)

        print("Loading nano-vllm engine (PagedAttention)...")
        from nano_vllm.engine import LLMEngine
        nano_engine_paged = LLMEngine(
            model_path=config.model_path,
            device=config.device,
            dtype=config.dtype,
            max_batch_size=max(config.batch_sizes),
            use_paged_attention=True,
        )
        print("nano-vllm (PagedAttention) loaded.")

        for idx, (batch_size, max_tokens, prompt_type) in enumerate(test_configs, 1):
            prompts = PROMPTS[prompt_type][:batch_size]
            print(f"\n[{idx}/{total_configs}] batch={batch_size}, "
                  f"max_tokens={max_tokens}, prompts={prompt_type}")

            reset_gpu_memory()

            # Warmup
            for _ in range(config.warmup_runs):
                benchmark_nano_vllm(nano_engine_paged, prompts, max_tokens)

            # Timed runs
            times = []
            tokens = []
            for _ in range(config.num_runs):
                reset_gpu_memory()
                t, tok = benchmark_nano_vllm(nano_engine_paged, prompts, max_tokens)
                times.append(t)
                tokens.append(tok)

            avg_time = sum(times) / len(times)
            avg_tokens = sum(tokens) / len(tokens)
            peak_mem = get_gpu_memory_mb()

            results.append(BenchmarkResult(
                backend="nano_vllm_paged",
                batch_size=batch_size,
                max_tokens=max_tokens,
                prompt_type=prompt_type,
                num_prompts=len(prompts),
                total_time_sec=avg_time,
                tokens_generated=int(avg_tokens),
                throughput_tokens_per_sec=avg_tokens / avg_time,
                avg_latency_per_prompt_ms=(avg_time / len(prompts)) * 1000,
                peak_memory_mb=peak_mem,
            ))
            print(f"  nano-vllm (paged): {avg_tokens/avg_time:.1f} tok/s, {avg_time:.3f}s")

        # Unload nano-vllm paged engine before loading next
        print("\nUnloading nano-vllm (PagedAttention) engine...")
        unload_model(nano_engine_paged)

    # ========== Benchmark nano-vllm legacy ==========
    if not skip_nano_legacy:
        print("\n" + "=" * 60)
        print("BENCHMARKING: nano-vllm (Legacy)")
        print("=" * 60)

        print("Loading nano-vllm engine (Legacy)...")
        from nano_vllm.engine import LLMEngine
        nano_engine_legacy = LLMEngine(
            model_path=config.model_path,
            device=config.device,
            dtype=config.dtype,
            max_batch_size=max(config.batch_sizes),
            use_paged_attention=False,
        )
        print("nano-vllm (Legacy) loaded.")

        for idx, (batch_size, max_tokens, prompt_type) in enumerate(test_configs, 1):
            prompts = PROMPTS[prompt_type][:batch_size]
            print(f"\n[{idx}/{total_configs}] batch={batch_size}, "
                  f"max_tokens={max_tokens}, prompts={prompt_type}")

            reset_gpu_memory()

            # Warmup
            for _ in range(config.warmup_runs):
                benchmark_nano_vllm(nano_engine_legacy, prompts, max_tokens)

            # Timed runs
            times = []
            tokens = []
            for _ in range(config.num_runs):
                reset_gpu_memory()
                t, tok = benchmark_nano_vllm(nano_engine_legacy, prompts, max_tokens)
                times.append(t)
                tokens.append(tok)

            avg_time = sum(times) / len(times)
            avg_tokens = sum(tokens) / len(tokens)
            peak_mem = get_gpu_memory_mb()

            results.append(BenchmarkResult(
                backend="nano_vllm_legacy",
                batch_size=batch_size,
                max_tokens=max_tokens,
                prompt_type=prompt_type,
                num_prompts=len(prompts),
                total_time_sec=avg_time,
                tokens_generated=int(avg_tokens),
                throughput_tokens_per_sec=avg_tokens / avg_time,
                avg_latency_per_prompt_ms=(avg_time / len(prompts)) * 1000,
                peak_memory_mb=peak_mem,
            ))
            print(f"  nano-vllm (legacy): {avg_tokens/avg_time:.1f} tok/s, {avg_time:.3f}s")

        # Unload nano-vllm legacy engine
        print("\nUnloading nano-vllm (Legacy) engine...")
        unload_model(nano_engine_legacy)

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    # Group results by configuration
    configs = {}
    for r in results:
        key = (r.batch_size, r.max_tokens, r.prompt_type)
        if key not in configs:
            configs[key] = {}
        configs[key][r.backend] = r

    # Print header
    print(f"\n{'Config':<35} | {'HuggingFace':<20} | {'nano-vllm (paged)':<20} | {'nano-vllm (legacy)':<20}")
    print(f"{'(batch, tokens, prompt)':<35} | {'tok/s':<20} | {'tok/s (speedup)':<20} | {'tok/s (speedup)':<20}")
    print("-" * 100)

    # Print each configuration
    for (batch_size, max_tokens, prompt_type), backends in sorted(configs.items()):
        config_str = f"({batch_size}, {max_tokens}, {prompt_type})"

        hf_throughput = backends.get("huggingface")
        hf_str = f"{hf_throughput.throughput_tokens_per_sec:.1f}" if hf_throughput else "N/A"

        paged = backends.get("nano_vllm_paged")
        if paged and hf_throughput:
            speedup = paged.throughput_tokens_per_sec / hf_throughput.throughput_tokens_per_sec
            paged_str = f"{paged.throughput_tokens_per_sec:.1f} ({speedup:.2f}x)"
        elif paged:
            paged_str = f"{paged.throughput_tokens_per_sec:.1f}"
        else:
            paged_str = "N/A"

        legacy = backends.get("nano_vllm_legacy")
        if legacy and hf_throughput:
            speedup = legacy.throughput_tokens_per_sec / hf_throughput.throughput_tokens_per_sec
            legacy_str = f"{legacy.throughput_tokens_per_sec:.1f} ({speedup:.2f}x)"
        elif legacy:
            legacy_str = f"{legacy.throughput_tokens_per_sec:.1f}"
        else:
            legacy_str = "N/A"

        print(f"{config_str:<35} | {hf_str:<20} | {paged_str:<20} | {legacy_str:<20}")

    print("=" * 100)

    # Print summary statistics
    print("\nSUMMARY")
    print("-" * 50)

    for backend in ["huggingface", "nano_vllm_paged", "nano_vllm_legacy"]:
        backend_results = [r for r in results if r.backend == backend]
        if backend_results:
            avg_throughput = sum(r.throughput_tokens_per_sec for r in backend_results) / len(backend_results)
            avg_memory = sum(r.peak_memory_mb for r in backend_results if r.peak_memory_mb) / len(backend_results)
            print(f"{backend}:")
            print(f"  Avg throughput: {avg_throughput:.1f} tokens/sec")
            if avg_memory > 0:
                print(f"  Avg peak memory: {avg_memory:.1f} MB")


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file."""
    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark nano-vllm vs HuggingFace")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--max-tokens",
        type=str,
        default="20,50,100",
        help="Comma-separated max tokens to test",
    )
    parser.add_argument(
        "--prompt-types",
        type=str,
        default="short,medium,long",
        help="Comma-separated prompt types (short/medium/long/extreme)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of timed runs to average",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HuggingFace benchmarks",
    )
    parser.add_argument(
        "--skip-paged",
        action="store_true",
        help="Skip nano-vllm PagedAttention benchmarks",
    )
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
        help="Skip nano-vllm legacy benchmarks",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer configurations for fast testing",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all available GPUs with device_map='auto' (HuggingFace only)",
    )
    parser.add_argument(
        "--extreme",
        action="store_true",
        help="Extreme mode: test with very long prompts (500-800 tokens each)",
    )

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # Parse configuration
    if args.extreme:
        # Long context testing mode
        batch_sizes = [1, 2]
        max_tokens_list = [50, 100]
        prompt_types = ["extreme"]
    elif args.quick:
        batch_sizes = [1, 4]
        max_tokens_list = [20, 50]
        prompt_types = ["short", "medium"]
    else:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        max_tokens_list = [int(x) for x in args.max_tokens.split(",")]
        prompt_types = args.prompt_types.split(",")

    config = BenchmarkConfig(
        model_path=args.model,
        batch_sizes=batch_sizes,
        max_tokens_list=max_tokens_list,
        prompt_types=prompt_types,
        device=args.device,
        dtype=dtype_map[args.dtype],
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        multi_gpu=args.multi_gpu,
    )

    print("=" * 60)
    print("BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Device: {config.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Multi-GPU: {config.multi_gpu}")
    if config.multi_gpu and torch.cuda.is_available():
        print(f"  Available GPUs: {torch.cuda.device_count()}")
    print(f"Batch sizes: {config.batch_sizes}")
    print(f"Max tokens: {config.max_tokens_list}")
    print(f"Prompt types: {config.prompt_types}")
    print(f"Warmup runs: {config.warmup_runs}")
    print(f"Timed runs: {config.num_runs}")
    print("=" * 60)

    # Run benchmarks
    results = run_benchmark(
        config,
        skip_hf=args.skip_hf,
        skip_nano_paged=args.skip_paged,
        skip_nano_legacy=args.skip_legacy,
    )

    # Print and save results
    print_results_table(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
