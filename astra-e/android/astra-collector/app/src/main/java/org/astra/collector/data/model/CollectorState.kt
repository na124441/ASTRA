package org.astra.collector.data.model

/**
 * High-reliability state machine for ASTRA Collector terminal workflow.
 */
enum class CollectorState {
    ASSIGNED,       // Task details loaded, operator reviewing protocol
    READY,          // Operator tapped "Start Recording", camera warming up
    RECORDING,      // Active video stream capture to private temp file
    RECORDED,       // Capture stopped, local MP4 ready for operator review
    UPLOADING,      // Resumable chunk streaming to ASTRA Upload API
    VERIFYING,      // Server-side assembly and Hugging Face hash verification
    COMPLETED,      // Upload verified; local storage verified clean; ready for next task
    UPLOAD_FAILED,  // Upload or verification error; local video retained
    RETRYING        // Automatic or manual upload retry in progress
}
