package org.astra.collector.data.model

import com.google.gson.annotations.SerializedName

data class CollectionTask(
    @SerializedName("task_id") val taskId: String,
    @SerializedName("experiment_id") val experimentId: String = "EXP001",
    @SerializedName("run_id") val runId: String,
    @SerializedName("camera_id") val cameraId: String,
    @SerializedName("scenario_type") val scenarioType: String,
    @SerializedName("required_object") val requiredObject: String,
    @SerializedName("target") val target: String,
    @SerializedName("duration_min") val durationMin: Int = 30,
    @SerializedName("duration_max") val durationMax: Int = 60,
    @SerializedName("orientation") val orientation: String = "landscape",
    @SerializedName("instruction_version") val instructionVersion: String = "EXP001-v1.0",
    @SerializedName("procedure_steps") val procedureSteps: List<String> = emptyList(),
    @SerializedName("status") val status: String = "available"
)

data class RecordingMetadata(
    @SerializedName("schema_version") val schemaVersion: String = "1.0",
    @SerializedName("experiment_id") val experimentId: String = "EXP001",
    @SerializedName("run_id") val runId: String,
    @SerializedName("recording_id") val recordingId: String,
    @SerializedName("collector_id") val collectorId: String,
    @SerializedName("camera_id") val cameraId: String,
    @SerializedName("scenario_type") val scenarioType: String,
    @SerializedName("object") val objectName: String,
    @SerializedName("target") val target: String,
    @SerializedName("duration_seconds") val durationSeconds: Double,
    @SerializedName("width") val width: Int = 1920,
    @SerializedName("height") val height: Int = 1080,
    @SerializedName("fps") val fps: Double = 30.0,
    @SerializedName("orientation") val orientation: String = "landscape",
    @SerializedName("file_size_bytes") val fileSizeBytes: Long,
    @SerializedName("sha256") val sha256: String,
    @SerializedName("app_version") val appVersion: String = "0.1.0",
    @SerializedName("protocol_version") val protocolVersion: String = "EXP001-v1.0",
    @SerializedName("created_at") val createdAt: String
)

data class DeviceRegisterRequest(
    @SerializedName("collector_id") val collectorId: String,
    @SerializedName("device_id") val deviceId: String,
    @SerializedName("app_version") val appVersion: String,
    @SerializedName("device_model") val deviceModel: String? = null
)

data class DeviceAuthResponse(
    @SerializedName("collector_id") val collectorId: String,
    @SerializedName("auth_token") val authToken: String,
    @SerializedName("status") val status: String,
    @SerializedName("message") val message: String
)

data class UploadInitiateRequest(
    @SerializedName("task_id") val taskId: String,
    @SerializedName("collector_id") val collectorId: String,
    @SerializedName("file_size_bytes") val fileSizeBytes: Long,
    @SerializedName("total_chunks") val totalChunks: Int,
    @SerializedName("sha256") val sha256: String,
    @SerializedName("metadata") val metadata: RecordingMetadata
)

data class UploadInitiateResponse(
    @SerializedName("upload_id") val uploadId: String,
    @SerializedName("chunk_size_bytes") val chunkSizeBytes: Int,
    @SerializedName("total_chunks") val totalChunks: Int,
    @SerializedName("status") val status: String
)

data class ChunkUploadResult(
    @SerializedName("upload_id") val uploadId: String,
    @SerializedName("chunk_index") val chunkIndex: Int,
    @SerializedName("bytes_received") val bytesReceived: Long,
    @SerializedName("chunk_sha256") val chunkSha256: String,
    @SerializedName("chunks_completed") val chunksCompleted: Int,
    @SerializedName("total_chunks") val totalChunks: Int
)

data class UploadCompleteRequest(
    @SerializedName("upload_id") val uploadId: String,
    @SerializedName("expected_sha256") val expectedSha256: String,
    @SerializedName("metadata") val metadata: RecordingMetadata
)

data class UploadStatusResponse(
    @SerializedName("upload_id") val uploadId: String,
    @SerializedName("task_id") val taskId: String,
    @SerializedName("status") val status: String,
    @SerializedName("verified") val verified: Boolean,
    @SerializedName("remote_path") val remotePath: String? = null,
    @SerializedName("sha256_match") val sha256Match: Boolean = false,
    @SerializedName("error_message") val errorMessage: String? = null
)
