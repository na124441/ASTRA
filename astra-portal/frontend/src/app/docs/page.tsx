/**
 * ============================================================================
 * OWNER: Frontend Developer 2
 * PURPOSE: Interactive Model Execution Guide & Documentation.
 * ============================================================================
 */

import { CodeBlock } from '@/components/code-block';
import { BookOpen, Terminal, CheckCircle2, ShieldAlert } from 'lucide-react';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
export default function DocsPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-16">
      {/* Header */}
      <div className="mb-12">
        <div className="flex items-center gap-2 text-xs font-mono text-cyan-accent uppercase tracking-widest mb-2">
          <BookOpen className="h-4 w-4" />
          <span>Standard Operating Procedure</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white">
          ASTRA-E Model Run & Deployment Guide
        </h1>
        <p className="mt-3 text-sm text-text-secondary leading-relaxed">
          How to configure your local edge environment, download quantized action recognition checkpoints, and run real-time camera inference for experimental procedure validation.
        </p>
      </div>

      {/* Step 1 */}
      <section className="mb-12">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-cyan-dim/30 border border-cyan-accent text-cyan-accent text-xs font-mono">1</span>
          <span>Environment Setup & Installation</span>
        </h2>
        <p className="mt-2 text-xs text-text-secondary">
          Ensure Python 3.10+ is installed on your system. Create an isolated virtual environment and install ONNX Runtime.
        </p>

        <CodeBlock
          title="Step 1: Setup Python Virtual Environment"
          powershellCode={`# Create and activate virtual environment
python -m venv astra-env
.\astra-env\Scripts\activate

# Install inference dependencies
pip install onnxruntime opencv-python-headless numpy requests`}
          bashCode={`# Create and activate virtual environment
python3 -m venv astra-env
source astra-env/bin/activate

# Install inference dependencies
pip install onnxruntime opencv-python-headless numpy requests`}
        />
      </section>

      {/* Step 2 */}
      <section className="mb-12">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-cyan-dim/30 border border-cyan-accent text-cyan-accent text-xs font-mono">2</span>
          <span>Download Quantized Weights</span>
        </h2>
        <p className="mt-2 text-xs text-text-secondary">
          Download the INT8 quantized model checkpoint (`exp001-int8.onnx`, 142 MB) from our private Hugging Face release.
        </p>

        <CodeBlock
          title="Step 2: Fetch ONNX Checkpoint"
          powershellCode={`# Download weights using PowerShell
Invoke-WebRequest -Uri "https://huggingface.co/na124441/astra-e-raw/resolve/main/models/exp001-int8.onnx" -OutFile "exp001-int8.onnx"`}
bashCode={`# Download weights using curl
curl -L -o exp001-int8.onnx "https://huggingface.co/na124441/astra-e-raw/resolve/main/models/exp001-int8.onnx"`}
        />
      </section>

      {/* Step 3 */}
      <section className="mb-12">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-cyan-dim/30 border border-cyan-accent text-cyan-accent text-xs font-mono">3</span>
          <span>Execute Real-Time Camera Inference</span>
        </h2>
        <p className="mt-2 text-xs text-text-secondary">
          Run inference against your primary webcam or a pre-recorded EXP001 video file.
        </p>

        <CodeBlock
          title="Step 3: Run Inference Script"
          powershellCode={`# Run inference on default webcam (Device 0)
python infer.py --model exp001-int8.onnx --camera 0 --protocol EXP001

# Or run inference on a sample video file
python infer.py --model exp001-int8.onnx --video sample_run.mp4`}
          bashCode={`# Run inference on default webcam (Device 0)
python3 infer.py --model exp001-int8.onnx --camera 0 --protocol EXP001

# Or run inference on a sample video file
python3 infer.py --model exp001-int8.onnx --video sample_run.mp4`}
        />
      </section>
    

      {/* Troubleshooting Section */}
      <section className="p-6 rounded-2xl border border-space-border bg-space-card/30 mt-8">
        <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-3">
          <ShieldAlert className="h-4 w-4 text-amber-accent" />
          <span>Troubleshooting & Common Invariants</span>
        </h3>
        <ul className="space-y-2 text-xs text-text-secondary">
          <li>• <strong className="text-white">Camera Access Error:</strong> Ensure no other application (Zoom, Teams, Chrome) is holding the webcam lock.</li>
          <li>• <strong className="text-white">ONNX Runtime GPU:</strong> For NVIDIA CUDA acceleration, run <code>pip install onnxruntime-gpu</code>.</li>
          <li>• <strong className="text-white">Checksum Discrepancy:</strong> Verify your file SHA-256 against our official release table on the Downloads page.</li>
        </ul>
      </section>
      {/* FAQ Section */}
<section className="mt-12">
  <h2 className="text-xl font-bold text-white mb-6">
    Frequently Asked Questions
  </h2>

  <Accordion type="single" collapsible className="w-full">
    <AccordionItem value="camera">
      <AccordionTrigger className="text-sm text-white">
        What should I do if I get a camera permission error?
      </AccordionTrigger>
      <AccordionContent className="text-xs text-text-secondary">
        Make sure your webcam is connected and no other application such as
        Zoom, Teams, or Chrome is currently using it. On Windows, also check
        Camera Privacy settings and allow access to desktop applications.
      </AccordionContent>
    </AccordionItem>

    <AccordionItem value="gpu">
      <AccordionTrigger className="text-sm text-white">
        How can I fix a GPU Out of Memory error?
      </AccordionTrigger>
      <AccordionContent className="text-xs text-text-secondary">
        Try using the INT8 quantized model, reduce the input resolution, or
        close other GPU-intensive applications. If GPU inference is not
        required, use CPU inference instead.
      </AccordionContent>
    </AccordionItem>

    <AccordionItem value="checksum">
      <AccordionTrigger className="text-sm text-white">
        What does a checksum mismatch mean?
      </AccordionTrigger>
      <AccordionContent className="text-xs text-text-secondary">
        The downloaded model may be incomplete or corrupted. Re-download the
        model and compare its SHA-256 hash with the official checksum shown on
        the Downloads page.
      </AccordionContent>
    </AccordionItem>

    <AccordionItem value="python">
      <AccordionTrigger className="text-sm text-white">
        What should I do if I encounter a Python dependency error?
      </AccordionTrigger>
      <AccordionContent className="text-xs text-text-secondary">
        Make sure the astra-env virtual environment is activated and install
        all required dependencies using the commands provided in Step 1.
      </AccordionContent>
    </AccordionItem>
  </Accordion>
</section>
    </div>
  );
}
