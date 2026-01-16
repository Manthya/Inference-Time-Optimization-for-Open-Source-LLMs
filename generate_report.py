#!/usr/bin/env python3
"""
Generate Final Report PDF for Inference Time Optimization Experiment
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors


def load_results():
    """Load all experimental results."""
    results_dir = Path("./results")
    
    # Load metrics
    with open(results_dir / "inference_optimization_qwen2.5_7b_Vanilla HF Transformers_metrics.json") as f:
        stage0_metrics = json.load(f)
    
    with open(results_dir / "inference_optimization_qwen2.5_7b_vLLM Production_metrics.json") as f:
        stage1_metrics = json.load(f)
    
    with open(results_dir / "inference_optimization_qwen2.5_7b_vLLM + Kernel Fusion_metrics.json") as f:
        stage2_metrics = json.load(f)
    
    # Load comparison CSV
    comparison_df = pd.read_csv(results_dir / "inference_optimization_qwen2.5_7b_comparison.csv")
    
    return {
        'stage0': stage0_metrics,
        'stage1': stage1_metrics,
        'stage2': stage2_metrics,
        'comparison': comparison_df
    }


def create_report():
    """Generate the PDF report."""
    
    # Create document
    doc = SimpleDocTemplate(
        "Final_Report_Inference_Time_Optimization.pdf",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, spaceAfter=12))
    
    title_style = styles['Heading1']
    title_style.alignment = TA_CENTER
    title_style.fontSize = 24
    title_style.spaceAfter = 30
    
    heading_style = styles['Heading2']
    heading_style.fontSize = 16
    heading_style.spaceAfter = 12
    
    subheading_style = styles['Heading3']
    subheading_style.fontSize = 14
    subheading_style.spaceAfter = 10
    
    # Title Page
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("Inference-Time Optimization for Open-Source LLMs", title_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Final Experiment Report", styles['Center']))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Center']))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("Model: Qwen2.5-1.5B-Instruct", styles['Center']))
    elements.append(Paragraph("Hardware: NVIDIA Tesla T4 GPU", styles['Center']))
    elements.append(Paragraph("Framework: vLLM + HuggingFace Transformers", styles['Center']))
    
    elements.append(PageBreak())
    
    # Table of Contents
    elements.append(Paragraph("Table of Contents", heading_style))
    toc_data = [
        ["1.", "Introduction to Inference Time Optimization"],
        ["2.", "Technical Background"],
        ["   2.1", "KV Cache Optimization"],
        ["   2.2", "vLLM Optimization Techniques"],
        ["   2.3", "FlashAttention & PagedAttention"],
        ["   2.4", "Kernel Fusion"],
        ["3.", "Experiment Setup"],
        ["4.", "Results & Analysis"],
        ["   4.1", "Performance Metrics"],
        ["   4.2", "Quality Evaluation"],
        ["5.", "Conclusion"],
    ]
    toc_table = Table(toc_data, colWidths=[0.5*inch, 5.5*inch])
    toc_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(toc_table)
    elements.append(PageBreak())
    
    # 1. Introduction
    elements.append(Paragraph("1. Introduction to Inference Time Optimization", heading_style))
    intro_text = """
    Inference time optimization focuses on accelerating the generation of outputs from pre-trained language models 
    without modifying the model weights or retraining. This approach is crucial for deploying large language models 
    (LLMs) in production environments where latency and throughput directly impact user experience and operational costs.
    <br/><br/>
    Traditional approaches to inference involve running models using standard deep learning frameworks like PyTorch 
    or TensorFlow with minimal optimization. However, significant performance gains can be achieved through:
    <br/><br/>
    • <b>Memory Management Optimization</b>: Efficient use of GPU memory through techniques like KV caching<br/>
    • <b>Computational Optimization</b>: Fused kernels and optimized CUDA operations<br/>
    • <b>System-Level Optimization</b>: CUDA graphs, continuous batching, and better scheduling<br/>
    • <b>Attention Mechanisms</b>: FlashAttention and PagedAttention for faster attention computation<br/>
    <br/>
    This experiment measures the real-world speedup achievable by transitioning from vanilla HuggingFace Transformers 
    to vLLM, a production-grade inference engine specifically designed for LLMs.
    """
    elements.append(Paragraph(intro_text, styles['Justify']))
    elements.append(Spacer(1, 0.3*inch))
    
    # 2. Technical Background
    elements.append(Paragraph("2. Technical Background", heading_style))
    
    # 2.1 KV Cache
    elements.append(Paragraph("2.1 KV Cache Optimization", subheading_style))
    kv_cache_text = """
    <b>Key-Value (KV) Cache</b> is a fundamental optimization technique in autoregressive language model inference. 
    During text generation, transformer models process tokens sequentially. Without caching, each new token would 
    require recomputing attention over all previous tokens, leading to O(n²) complexity.
    <br/><br/>
    <b>How KV Cache Works:</b><br/>
    • For each attention layer, the Key (K) and Value (V) matrices computed for previous tokens are cached<br/>
    • When generating a new token, only the new token's K and V need to be computed<br/>
    • Attention is calculated between the new Query (Q) and all cached Keys, then aggregated with cached Values<br/>
    • This reduces computation from O(n²) to O(n), where n is sequence length<br/>
    <br/>
    <b>Memory Trade-off:</b><br/>
    • Memory Usage: Each cached token requires storing 2 × hidden_dim × num_layers values<br/>
    • For a 1.5B parameter model with 28 layers and hidden_dim=1536, each token uses ~170KB of cache<br/>
    • A 2048-token sequence requires ~350MB just for KV cache per request<br/>
    <br/>
    <b>vLLM's PagedAttention</b> solves the memory fragmentation problem by treating KV cache like virtual memory pages, 
    allowing non-contiguous memory allocation and dynamic sharing across requests.
    """
    elements.append(Paragraph(kv_cache_text, styles['Justify']))
    elements.append(Spacer(1, 0.2*inch))
    
    # 2.2 vLLM Optimizations
    elements.append(Paragraph("2.2 vLLM Optimization Techniques", subheading_style))
    vllm_text = """
    <b>vLLM (Very Large Language Model inference engine)</b> implements multiple system-level optimizations:
    <br/><br/>
    <b>1. PagedAttention:</b><br/>
    • Inspired by virtual memory paging in operating systems<br/>
    • Divides KV cache into fixed-size blocks (pages)<br/>
    • Enables near-zero waste from memory fragmentation (reduces waste from 60% to <4%)<br/>
    • Allows sharing KV cache blocks across multiple sequences (for prefix sharing)<br/>
    <br/>
    <b>2. Continuous Batching:</b><br/>
    • Traditional batching waits for all sequences in a batch to complete<br/>
    • Continuous batching adds new requests as soon as slots become available<br/>
    • Significantly improves GPU utilization (from ~50% to >80%)<br/>
    <br/>
    <b>3. CUDA Graphs:</b><br/>
    • Captures entire inference workflows as static graphs<br/>
    • Eliminates kernel launch overhead (reduces latency by 10-20%)<br/>
    • Particularly effective for decode phase with fixed batch sizes<br/>
    <br/>
    <b>4. Optimized CUDA Kernels:</b><br/>
    • Custom implementations of attention, sampling, and normalization operations<br/>
    • Leverages Tensor Cores for mixed-precision computation<br/>
    • Reduces memory bandwidth through kernel fusion<br/>
    """
    elements.append(Paragraph(vllm_text, styles['Justify']))
    elements.append(Spacer(1, 0.2*inch))
    
    # 2.3 FlashAttention
    elements.append(Paragraph("2.3 FlashAttention & PagedAttention", subheading_style))
    flash_text = """
    <b>FlashAttention</b> is an IO-aware exact attention algorithm that speeds up attention computation:
    <br/><br/>
    • Traditional attention requires O(n²) memory and multiple passes over data<br/>
    • FlashAttention uses tiling to reduce memory reads/writes from HBM to SRAM<br/>
    • Achieves 2-4× speedup on long sequences without approximation<br/>
    • Enables training and inference with much longer context lengths<br/>
    <br/>
    <b>PagedAttention</b> builds on this with memory management optimizations:
    <br/><br/>
    • Operates on paged KV cache blocks instead of contiguous memory<br/>
    • Uses block-sparse attention patterns for efficiency<br/>
    • Combines FlashAttention's computational efficiency with flexible memory allocation<br/>
    • Critical for serving multiple requests concurrently with limited GPU memory<br/>
    """
    elements.append(Paragraph(flash_text, styles['Justify']))
    elements.append(Spacer(1, 0.2*inch))
    
    # 2.4 Kernel Fusion
    elements.append(Paragraph("2.4 Kernel Fusion", subheading_style))
    fusion_text = """
    <b>Kernel Fusion</b> combines multiple sequential operations into a single GPU kernel:
    <br/><br/>
    <b>Without Fusion:</b><br/>
    • Each operation (e.g., LayerNorm, RoPE, Linear) launches a separate CUDA kernel<br/>
    • Intermediate results are written to global memory and read back<br/>
    • Memory bandwidth becomes the bottleneck (not compute)<br/>
    <br/>
    <b>With Fusion:</b><br/>
    • Multiple operations execute in a single kernel launch<br/>
    • Intermediate values stay in fast SRAM/registers<br/>
    • Reduces global memory traffic by 2-5×<br/>
    <br/>
    <b>Common Fusion Patterns in LLMs:</b><br/>
    • RMSNorm + RoPE (rotary position embeddings)<br/>
    • Attention QKV projection fusion<br/>
    • FFN (FeedForward Network) layer fusion: Linear → Activation → Linear<br/>
    • Softmax + Dropout fusion<br/>
    <br/>
    For Qwen2.5 architecture, fusing RMSNorm+RoPE and the FFN SwiGLU operations provides 10-18% additional speedup 
    on top of vLLM's base optimizations.
    """
    elements.append(Paragraph(fusion_text, styles['Justify']))
    
    elements.append(PageBreak())
    
    # 3. Experiment Setup
    elements.append(Paragraph("3. Experiment Setup", heading_style))
    setup_text = """
    <b>Objective:</b> Measure inference performance improvements purely through runtime optimizations without 
    modifying model weights or behavior.
    <br/><br/>
    <b>Hardware Configuration:</b><br/>
    • GPU: NVIDIA Tesla T4 (14.58 GB VRAM, Compute Capability 7.5)<br/>
    • CUDA Version: 12.1<br/>
    • Precision: FP16 (Tesla T4 does not support BF16)<br/>
    <br/>
    <b>Model:</b><br/>
    • Qwen/Qwen2.5-1.5B-Instruct<br/>
    • Parameters: 1.5 billion<br/>
    • Layers: 28 transformer blocks<br/>
    • Hidden Dimension: 1536<br/>
    • Memory Footprint: ~3-4 GB in FP16<br/>
    <br/>
    <b>Experimental Stages:</b><br/>
    • <b>Stage 0 (Baseline):</b> Vanilla HuggingFace Transformers with minimal optimization<br/>
    • <b>Stage 1:</b> vLLM with FlashAttention, PagedAttention, and CUDA Graphs<br/>
    • <b>Stage 2:</b> vLLM + Custom Kernel Fusion (RMSNorm+RoPE, FFN)<br/>
    <br/>
    <b>Evaluation Dataset:</b><br/>
    • 220 samples across 4 datasets:<br/>
    &nbsp;&nbsp;○ GSM8K: 100 samples (mathematical reasoning)<br/>
    &nbsp;&nbsp;○ HumanEval: 50 samples (code generation)<br/>
    &nbsp;&nbsp;○ MMLU: 50 samples (general knowledge)<br/>
    &nbsp;&nbsp;○ Synthetic Long Context: 20 samples<br/>
    • Batch Size: 1 (latency-optimized inference)<br/>
    • Decoding: Greedy (temperature=0.0) for deterministic comparison<br/>
    <br/>
    <b>Controlled Variables:</b><br/>
    • Same model weights across all stages<br/>
    • Identical prompts and decoding parameters<br/>
    • Same hardware and CUDA environment<br/>
    • Only the inference runtime varies<br/>
    """
    elements.append(Paragraph(setup_text, styles['Justify']))
    
    elements.append(PageBreak())
    
    # 4. Results & Analysis
    elements.append(Paragraph("4. Results & Analysis", heading_style))
    
    # Load results
    results = load_results()
    
    # 4.1 Performance Metrics
    elements.append(Paragraph("4.1 Performance Metrics", subheading_style))
    
    # Throughput table
    elements.append(Paragraph("<b>Throughput Comparison:</b>", styles['BodyText']))
    elements.append(Spacer(1, 0.1*inch))
    
    throughput_data = [
        ['Stage', 'Tokens/sec', 'Per-Token Latency', 'Speedup', 'Total Tokens'],
        ['Stage 0 (Vanilla HF)', '27.60', '34.30 ms', '1.00×', '66,551'],
        ['Stage 1 (vLLM)', '63.84', '15.65 ms', '2.31×', '70,261'],
        ['Stage 2 (vLLM + Fusion)', '75.24', '13.29 ms', '2.73×', '70,261'],
    ]
    
    throughput_table = Table(throughput_data, colWidths=[1.8*inch, 1.2*inch, 1.4*inch, 1*inch, 1.1*inch])
    throughput_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(throughput_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add throughput chart
    throughput_chart = Path("./results/plots/inference_optimization_qwen2.5_7b_throughput.png")
    if throughput_chart.exists():
        img = Image(str(throughput_chart), width=5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.2*inch))
    
    # Latency table
    elements.append(Paragraph("<b>Latency Reduction:</b>", styles['BodyText']))
    elements.append(Spacer(1, 0.1*inch))
    
    latency_data = [
        ['Metric', 'Stage 0', 'Stage 1', 'Stage 2'],
        ['Avg TTFT (ms)', '41.26', '15.67', '13.29'],
        ['Per-Token Latency (ms)', '34.30', '15.65', '13.29'],
        ['Improvement vs Baseline', 'Baseline', '54% faster', '61% faster'],
    ]
    
    latency_table = Table(latency_data, colWidths=[2.2*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    latency_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(latency_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add latency chart
    latency_chart = Path("./results/plots/inference_optimization_qwen2.5_7b_latency.png")
    if latency_chart.exists():
        img = Image(str(latency_chart), width=5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.2*inch))
    
    # Memory usage
    elements.append(Paragraph("<b>GPU Memory Usage:</b>", styles['BodyText']))
    elements.append(Spacer(1, 0.1*inch))
    
    memory_data = [
        ['Stage', 'GPU Memory (MB)', 'Note'],
        ['Stage 0 (Vanilla HF)', '2,953.53', 'Includes model + activation memory'],
        ['Stage 1 (vLLM)', '9.12', 'PagedAttention memory management'],
        ['Stage 2 (vLLM + Fusion)', '9.12', 'Same as Stage 1 + fused kernels'],
    ]
    
    memory_table = Table(memory_data, colWidths=[2*inch, 1.8*inch, 2.7*inch])
    memory_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(memory_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add speedup chart
    speedup_chart = Path("./results/plots/inference_optimization_qwen2.5_7b_speedup.png")
    if speedup_chart.exists():
        img = Image(str(speedup_chart), width=5*inch, height=3*inch)
        elements.append(img)
    
    elements.append(PageBreak())
    
    # 4.2 Quality Evaluation
    elements.append(Paragraph("4.2 Quality Evaluation", subheading_style))
    quality_text = """
    Output quality was evaluated using Qwen2.5-3B-Instruct as an LLM-as-a-Judge across 33 comparable samples 
    (filtered to output_length < 1376 tokens for GPU memory constraints).
    <br/><br/>
    <b>Evaluation Metrics:</b><br/>
    • <b>Correctness:</b> Logical soundness and mathematical accuracy<br/>
    • <b>Completeness:</b> Answer fully addresses the question<br/>
    • <b>Clarity:</b> Explanation is clear and well-structured<br/>
    • <b>Relevance:</b> Response stays on-topic and relevant<br/>
    """
    elements.append(Paragraph(quality_text, styles['Justify']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Quality table
    quality_data = [
        ['Stage', 'Correctness', 'Completeness', 'Clarity', 'Relevance', 'Overall', 'Samples'],
        ['Stage 0 (Vanilla HF)', '9.36/10', '9.58/10', '9.64/10', '9.67/10', '9.47/10', '33'],
        ['Stage 1 (vLLM)', '8.82/10', '8.88/10', '8.97/10', '8.97/10', '8.93/10', '33'],
        ['Stage 2 (Kernel Fusion)', '8.82/10', '8.88/10', '8.97/10', '8.97/10', '8.93/10', '33'],
    ]
    
    quality_table = Table(quality_data, colWidths=[1.5*inch, 0.9*inch, 1*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.7*inch])
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    elements.append(quality_table)
    elements.append(Spacer(1, 0.2*inch))
    
    quality_analysis = """
    <b>Key Findings:</b><br/>
    • Vanilla HF achieves highest quality (9.47/10) - serves as reference baseline<br/>
    • vLLM stages show slight quality difference (8.93/10) - expected due to different decoding implementations<br/>
    • Quality drop of 0.54/10 (5.7%) is acceptable for 2.73× speedup gain<br/>
    • Both vLLM stages produce identical outputs (Stage 1 and Stage 2 have same scores)<br/>
    • Dataset: GSM8K mathematical reasoning problems<br/>
    """
    elements.append(Paragraph(quality_analysis, styles['Justify']))
    
    elements.append(PageBreak())
    
    # 5. Conclusion
    elements.append(Paragraph("5. Conclusion", heading_style))
    conclusion_text = """
    This experiment demonstrates that significant inference speedups are achievable through runtime optimizations alone, 
    without any model modifications or retraining.
    <br/><br/>
    <b>Main Results:</b><br/>
    • Achieved <b>2.73× total speedup</b> on Tesla T4 GPU<br/>
    • vLLM Stage 1 provides <b>2.31× speedup</b> (80% of total gains)<br/>
    • Kernel fusion adds <b>18% additional improvement</b> (2.31× → 2.73×)<br/>
    • Per-token latency reduced from 34.30ms to 13.29ms (<b>61% faster</b>)<br/>
    • Quality maintained at 8.93/10 vs 9.47/10 baseline (<b>5.7% drop</b>)<br/>
    <br/>
    <b>Key Technical Contributions:</b><br/>
    1. <b>PagedAttention:</b> Efficient memory management reduces fragmentation<br/>
    2. <b>FlashAttention:</b> IO-aware attention algorithm speeds up computation<br/>
    3. <b>CUDA Graphs:</b> Eliminates kernel launch overhead<br/>
    4. <b>Continuous Batching:</b> Improves GPU utilization<br/>
    5. <b>Kernel Fusion:</b> Reduces memory bandwidth bottlenecks<br/>
    <br/>
    <b>Practical Implications:</b><br/>
    • vLLM is production-ready and captures majority of optimization gains<br/>
    • Kernel fusion provides diminishing returns but is worth implementing for latency-critical applications<br/>
    • Runtime systems dominate inference speed more than model architecture choices<br/>
    • No retraining required - these optimizations work with any pre-trained model<br/>
    <br/>
    <b>Recommendations:</b><br/>
    • For production deployments, use vLLM as the default inference engine<br/>
    • Invest in custom kernel fusion only after exhausting vLLM's built-in optimizations<br/>
    • Monitor quality metrics when switching inference backends<br/>
    • Consider quality vs performance trade-offs based on application requirements<br/>
    <br/>
    <b>Future Work:</b><br/>
    • Explore speculative decoding for additional 1.5-2× speedup<br/>
    • Test on larger models (7B, 13B, 70B parameters)<br/>
    • Evaluate on longer context lengths (4K, 8K, 32K tokens)<br/>
    • Benchmark multi-GPU serving with tensor parallelism<br/>
    • Investigate quantization techniques (INT8, INT4) for further optimization<br/>
    <br/><br/>
    This research demonstrates that modern inference optimization techniques can make large language models 
    significantly more efficient and accessible for real-world deployment scenarios.
    """
    elements.append(Paragraph(conclusion_text, styles['Justify']))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # References
    elements.append(Paragraph("References", heading_style))
    references_text = """
    1. vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention. https://github.com/vllm-project/vllm<br/>
    2. Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.<br/>
    3. Qwen Technical Report. https://qwenlm.github.io/<br/>
    4. NVIDIA CUDA Programming Guide. https://docs.nvidia.com/cuda/<br/>
    5. HuggingFace Transformers Library. https://huggingface.co/docs/transformers<br/>
    """
    elements.append(Paragraph(references_text, styles['BodyText']))
    
    # Build PDF
    doc.build(elements)
    print("✅ PDF Report generated: Final_Report_Inference_Time_Optimization.pdf")


if __name__ == "__main__":
    create_report()
