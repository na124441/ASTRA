package org.astra.collector.worker

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.google.gson.Gson
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.astra.collector.data.local.AstraCollectorDatabase
import org.astra.collector.data.model.RecordingMetadata
import org.astra.collector.data.model.UploadCompleteRequest
import org.astra.collector.data.remote.CollectorApiService
import org.astra.collector.storage.LocalStorageManager
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

/**
 * Robust WorkManager worker that streams 8 MB chunks, verifies remote integrity,
 * and deletes local video ONLY upon cryptographic server confirmation.
 */
class UploadWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    private val tag = "AstraUploadWorker"
    private val storageManager = LocalStorageManager(context)
    private val db = AstraCollectorDatabase.getInstance(context)

    override suspend fun doWork(): Result {
        val uploadId = inputData.getString(KEY_UPLOAD_ID) ?: return Result.failure()
        val filePath = inputData.getString(KEY_FILE_PATH) ?: return Result.failure()
        val expectedSha256 = inputData.getString(KEY_SHA256) ?: return Result.failure()
        val metadataJson = inputData.getString(KEY_METADATA_JSON) ?: return Result.failure()
        val serverUrl = inputData.getString(KEY_SERVER_URL) ?: return Result.failure()

        val videoFile = File(filePath)
        if (!videoFile.exists() || videoFile.length() == 0L) {
            Log.e(tag, "Target video file missing or empty: $filePath")
            return Result.failure()
        }

        val metadata = try {
            Gson().fromJson(metadataJson, RecordingMetadata::class.java)
        } catch (e: Exception) {
            Log.e(tag, "Failed to parse metadata JSON", e)
            return Result.failure()
        }

        val api = CollectorApiService.create(serverUrl)
        val chunkSize = 8 * 1024 * 1024 // 8 MB
        val totalBytes = videoFile.length()
        val totalChunks = ((totalBytes + chunkSize - 1) / chunkSize).toInt()

        Log.i(tag, "Starting upload $uploadId for file ${videoFile.name} ($totalBytes bytes, $totalChunks chunks)")

        try {
            // 1. Stream each chunk sequentially
            FileInputStream(videoFile).use { fis ->
                val buffer = ByteArray(chunkSize)
                var chunkIdx = 0

                while (chunkIdx < totalChunks) {
                    val bytesRead = fis.read(buffer)
                    if (bytesRead <= 0) break

                    val chunkBytes = if (bytesRead == chunkSize) buffer else buffer.copyOf(bytesRead)
                    val chunkDigest = MessageDigest.getInstance("SHA-256")
                    val chunkSha256 = chunkDigest.digest(chunkBytes).joinToString("") { "%02x".format(it) }

                    val requestBody = chunkBytes.toRequestBody("application/octet-stream".toMediaTypeOrNull())

                    Log.d(tag, "Uploading chunk $chunkIdx/$totalChunks ($bytesRead bytes)...")
                    val chunkResult = api.uploadChunk(
                        uploadId = uploadId,
                        chunkIndex = chunkIdx,
                        chunkSha256 = chunkSha256,
                        chunkData = requestBody
                    )

                    val progressPercent = ((chunkIdx + 1) * 100) / totalChunks
                    setProgress(workDataOf(KEY_PROGRESS to progressPercent))

                    chunkIdx++
                }
            }

            // 2. Finalize and verify remote persistence
            Log.i(tag, "All chunks uploaded. Requesting server verification for $uploadId...")
            val completeRequest = UploadCompleteRequest(
                uploadId = uploadId,
                expectedSha256 = expectedSha256,
                metadata = metadata
            )

            val statusResponse = api.completeUpload(uploadId, completeRequest)

            // 3. HARD INVARIANT: Check verified status
            if (statusResponse.verified && statusResponse.status.equals("verified", ignoreCase = true)) {
                Log.i(tag, "Server confirmed VERIFIED upload at ${statusResponse.remotePath}. Deleting local file...")
                
                val deleted = storageManager.deleteVerifiedFile(
                    file = videoFile,
                    serverStatus = statusResponse.status,
                    isVerified = statusResponse.verified
                )

                if (deleted) {
                    Log.i(tag, "Verified cleanup complete for upload $uploadId.")
                    return Result.success(workDataOf(KEY_REMOTE_PATH to statusResponse.remotePath))
                } else {
                    Log.w(tag, "Upload verified, but local deletion pending confirmation.")
                    return Result.success()
                }
            } else {
                Log.e(tag, "Server verification failed: ${statusResponse.errorMessage}. Local file PRESERVED.")
                return Result.retry()
            }

        } catch (e: Exception) {
            Log.e(tag, "Upload worker encountered exception: ${e.message}. Retrying with exponential backoff...", e)
            return Result.retry()
        }
    }

    companion object {
        const val KEY_UPLOAD_ID = "upload_id"
        const val KEY_TASK_ID = "task_id"
        const val KEY_FILE_PATH = "file_path"
        const val KEY_SHA256 = "sha256"
        const val KEY_METADATA_JSON = "metadata_json"
        const val KEY_SERVER_URL = "server_url"
        const val KEY_PROGRESS = "progress"
        const val KEY_REMOTE_PATH = "remote_path"
    }
}
